#!/usr/bin/env python
from pyscf import gto, scf, dft, cc, solvent, mp
from pyscf.hessian import thermo
from pyscf.solvent import hsm
import numpy

mol = gto.M(
    atom = '''
  O          -0.00308160220954      0.40897504460660      0.00000000000000
  H           0.74912540253019     -0.19884520216613      0.00000000000000
  H          -0.74604379032065     -0.21012984244046      0.00000000000000
''',
    basis   = 'cc-pvtz',
    unit    = 'angstrom',
    verbose = 4,
)

pcm_obj = hsm.PCM(mol)
pcm_obj.cavity_coords = mol.atom_coords(unit='B')
pcm_obj.method        = 'C-PCM'
pcm_obj.eps           = 80.1510
pcm_obj.vdw_scale     = 1.4
pcm_obj.lebedev_order = 17

# Calculation level
# Hartree-Fock
#mymp = scf.RHF(mol).PCM(pcm_obj)
#mymp.kernel()
# DFT
#mymp = dft.RKS(mol,xc='b3lypg').PCM(pcm_obj)
#mymp.kernel()
# MP2
mf = scf.RHF(mol)
mf.kernel()
mymp = mp.MP2(mf).PCM(pcm_obj)
mymp.kernel()

def fd_hessian(mymp, step=5e-3):
    mol = mymp.mol
    g = mymp.nuc_grad_method()
    g.kernel()
    g_scan = g.as_scanner()

    pmol = mol.copy()
    pmol.build()

    coords0     = pmol.atom_coords(unit='B')
    natm        = pmol.natm
    hessian     = numpy.zeros([natm, natm, 3, 3])

    for i in range(natm):
        for j in range(3):
            disp = numpy.zeros_like(coords0)
            disp[i, j] = step

            pmol.set_geom_(coords0 + disp, unit='B')
            pmol.build()
            e0, g0 = g_scan(pmol)


            pmol.set_geom_(coords0 - disp, unit='B')
            pmol.build()
            e1, g1 = g_scan(pmol)

            g0 = numpy.asarray(g0)
            g1 = numpy.asarray(g1)

            hessian[i, :, j, :] = (g0 - g1)/(2.0 * step)

    return hessian

# Numerical hessian
hessian = fd_hessian(mymp, step=5e-3)
print(hessian)

# thermodynamics
results = thermo.harmonic_analysis(
    mol, hessian,
    exclude_trans=False,
    exclude_rot=False,
    imaginary_freq=False,
)

freqs = results['freq_wavenumber'].real
norm_mode = results['norm_mode']

natm = mol.natm
print("Frequencies [cm^-1]:")
for i, w in enumerate(freqs):
    print(f"{i}: {w:.4f}")

thermo.dump_normal_mode(mol, results)
