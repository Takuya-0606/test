#!/usr/bin/env python
from pyscf import gto, scf, dft, mp
from pyscf.solvent import pcm
import numpy
import thermo_fd

mol = gto.M(atom='''
  O        0.000000   -0.000000    0.119491
  H        0.000000   -0.752067   -0.477966
  H        0.000000    0.752067   -0.477966
''', unit='Ang', charge=0, spin=0, basis='cc-pvtz', verbose=4)
mol.build()

#HSM set up
cavity_coords = mol.atom_coords(unit='B')
pcm_obj = pcm.PCM(mol)
pcm_obj.cavity_coords = cavity_coords
pcm_obj.method        = 'C-PCM'
pcm_obj.eps           = 80.1510
pcm_obj.lebedev_order = 17

# Calculation level
# Hartree-Fock
#mf = scf.RHF(mol).PCM(pcm_obj)
# DFT
mf = dft.RKS(mol,xc='b3lypg-d3bj').PCM(pcm_obj)
mf.kernel()
# MP2
#mf = scf.RHF(mol)
#mf.kernel()
#mymp = mp.MP2(mf).PCM(pcm_obj)
#mymp.kernel()

def gradient_scanner(mf):
    grad_drv = mf.nuc_grad_method().as_scanner()
    pcm_obj  = getattr(mf, 'with_solvent', None)

    def grad_at(coords_bohr):
        displaced = mf.mol.copy()
        displaced.set_geom_(coords_bohr, unit='B')
        displaced.build(False, False)
        if pcm_obj is not None:
            pcm_obj.reset(displaced)
        energy, grad = grad_drv(displaced)
        return grad.reshape(-1)

    return grad_at

def fd_hessian(grad_fn, coords_bohr, step):
    coords      = numpy.array(coords_bohr, dtype=float)
    flat_coords = coords.reshape(-1)
    ndim        = flat_coords.size
    hessian     = numpy.zeros((ndim, ndim))

    for i in range(ndim):
        delta     = numpy.zeros(ndim)
        delta[i]  = step
        forward   = (flat_coords + delta).reshape(coords.shape)
        backward  = (flat_coords - delta).reshape(coords.shape)
        grad_for  = grad_fn(forward)
        grad_back = grad_fn(backward)
        hessian[:,i] = (grad_for - grad_back) / (2.0 * step)

    return hessian

# HF & DFT
grad_fn = gradient_scanner(mf)
# MP2
#grad_fn = gradient_scanner(mymp)
step    = 5e-3 #Bohr
hessian = fd_hessian(grad_fn, cavity_coords, step)
print(hessian)

# thermodynamics
mass   = mol.atom_mass_list(isotope_avg=True)
coords = mol.atom_coords(unit='B')
freq   = thermo_fd.harmonic_analysis(mol, hessian, imaginary_freq=True, exclude_trans=False, exclude_rot=False)
freqs_tr = thermo_fd.compute_tr_frequencies(hessian, mass, coords)
tr,vib,full,nTR = thermo_fd.collect_freq(mass, coords, freq, freqs_tr)

thermo_fd.show_frequencies(mass, coords, hessian, freq)
result = thermo_fd.calc_hsm(tr, vib, T=298.15)
thermo_fd.print_hsm_tables(result)
