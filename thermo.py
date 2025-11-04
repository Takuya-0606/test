#!/usr/bin/env python
#
# This code was copied from the data generation program of Tencent Alchemy
# project (https://github.com/tencent-alchemy).

#
# Copyright 2019 Tencent America LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
# Author: Takuya Hashimoto

'''
Thermochemistry analysis.

Ref:
    psi4/psi4/driver/qcdb/vib.py
    http://gaussian.com/vib/

2025/08/19:
    Add HSM tools
'''

from functools import reduce
import numpy
import math
from scipy.constants import R, k, N_A, h, c
from pyscf import lib
from pyscf.data import nist
from tabulate import tabulate

LINDEP_THRESHOLD = 1e-7
GREEN = "\033[32m"
RED   = "\033[31m"
RESET = "\033[0m"

def _print_list(title, freqs):
            GREEN = "\033[32m"
            RED   = "\033[31m"
            RESET = "\033[0m"
            print()
            print("Vibration frequency calculation (cm^-1)")
            print(title)
            print("-"*40)
            for i, f in enumerate(freqs):
                # numpy.complexfloating でも動くように lib.scimath ベースの判定
                if numpy.iscomplexobj(f) and getattr(f, 'imag', 0) != 0:
                    tag = f"{RED}Imag{RESET}"
                    val = f"{abs(numpy.imag(f)):9.2f} i   "
                else:
                    tag = f"{GREEN}Real{RESET}"
                    val = f"{numpy.real(f):9.2f}     "
                print(f"Mode {i+1:2d}: {val} [{tag}]")
            print("-"*40)
            print()


def harmonic_analysis(mol, hess, exclude_trans=True, exclude_rot=True,
                      imaginary_freq=True, all_freq=True, mass=None):
    '''Each column is one mode

    imaginary_freq (boolean): save imaginary_freq as complex number (if True)
    or negative real number (if False)
    '''
    if mass is None:
        mass = mol.atom_mass_list(isotope_avg=True)
    results = {}
    atom_coords = mol.atom_coords()
    mass_center = numpy.einsum('z,zx->x', mass, atom_coords) / mass.sum()
    atom_coords = atom_coords - mass_center
    natm = atom_coords.shape[0]

    hess = numpy.asarray(hess)
    if hess.ndim == 2:
        hess = hess.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)
    elif hess.shape != (natm, natm, 3, 3):
        raise ValueError(f"Unexpected hessian shape: {hess.shape} (expected (natm,natm,3,3) or (3N,3N))")

    mass_hess = numpy.einsum('pqxy,p,q->pqxy', hess, mass**-.5, mass**-.5)
    h = mass_hess.transpose(0,2,1,3).reshape(natm*3,natm*3)

    #print("---- mass-weighted hessian ----")
    #print(h)
    #print("-------------------------------")

    remake = numpy.zeros((natm*3, natm*3))
    for i in range(natm*3):
        for j in range(natm*3):
            remake[i,j] = h[i,j]
            remake[j,i] = h[i,j]
    
    #print("---- mass-weighted hessian ----")
    #print(remake)
    #print("-------------------------------")

    TR = _get_TR(mass, atom_coords)
    TRspace = []
    if exclude_trans:
        TRspace.append(TR[:3])

    if exclude_rot:
        rot_const = rotation_const(mass, atom_coords)
        rotor_type = _get_rotor_type(rot_const)
        if rotor_type == 'ATOM':
            pass
        elif rotor_type == 'LINEAR':  # linear molecule
            TRspace.append(TR[3:5])
        else:
            TRspace.append(TR[3:])

    if TRspace:
        TRspace = numpy.vstack(TRspace)
        q, r = numpy.linalg.qr(TRspace.T)
        P = numpy.eye(natm * 3) - q.dot(q.T)
        w, v = numpy.linalg.eigh(P)
        bvec = v[:,w > LINDEP_THRESHOLD]
        remake = reduce(numpy.dot, (bvec.T, remake, bvec))
        force_const_au, mode = numpy.linalg.eigh(remake)
        mode = bvec.dot(mode)
    else:
        force_const_au, mode = numpy.linalg.eigh(remake)

    freq_au = numpy.lib.scimath.sqrt(force_const_au)
    results['freq_error'] = numpy.count_nonzero(freq_au.imag > 0)
    if not imaginary_freq and numpy.iscomplexobj(freq_au):
        # save imaginary frequency as negative frequency
        freq_au = freq_au.real - abs(freq_au.imag)

    results['freq_au'] = freq_au
    au2hz = (nist.HARTREE2J / (nist.ATOMIC_MASS * nist.BOHR_SI**2))**.5 / (2 * numpy.pi)
    freq_wavenumber = freq_au * au2hz / nist.LIGHT_SPEED_SI * 1e-2
    results['freq_wavenumber'] = freq_au * au2hz / nist.LIGHT_SPEED_SI * 1e-2

    norm_mode = numpy.einsum('z,zri->izr', mass**-.5, mode.reshape(natm,3,-1))
    results['norm_mode'] = norm_mode
    reduced_mass = 1./numpy.einsum('izr,izr->i', norm_mode, norm_mode)
    results['reduced_mass'] = reduced_mass

    # https://en.wikipedia.org/wiki/Vibrational_temperature
    results['vib_temperature'] = freq_au * au2hz * nist.PLANCK / nist.BOLTZMANN

    # force constants
    dyne = 1e-2 * nist.HARTREE2J / nist.BOHR_SI**2
    results['force_const_au'] = force_const_au
    results['force_const_dyne'] = reduced_mass * force_const_au * dyne  #cm^-1/a0^2

    if all_freq:
        _print_list("w/o projection", results['freq_wavenumber'])

    #TODO: IR intensity
    return results

def compute_tr_frequencies(hess, mass, coords):
    import numpy as np
    natm = len(mass)

    hess = np.asarray(hess)
    if hess.ndim == 2:
        h4 = hess.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)
    else:
        h4 = hess
    H = h4.transpose(0, 2, 1, 3).reshape(3*natm, 3*natm)
    
    remake = numpy.zeros((natm*3, natm*3))
    for i in range(natm*3):
        for j in range(natm*3):
            remake[i,j] = H[i,j]
            remake[j,i] = H[i,j]
    
    # mass-weighted Hessian
    mhalf = np.repeat(mass**-0.5, 3)
    H_mass = remake * mhalf[:,None] * mhalf[None,:]

    # TR vector
    TR = np.vstack(_get_TR(mass, coords))

    # Normalization
    TR = TR / np.linalg.norm(TR, axis=1)[:, None]

    # force constant
    force_consts = np.array([v @ H_mass @ v for v in TR])

    # frequency
    from pyscf.data import nist
    au2hz = (nist.HARTREE2J / (nist.ATOMIC_MASS * nist.BOHR_SI**2))**.5 / (2 * np.pi)
    freq_cm1 = np.sqrt(np.abs(force_consts)) * au2hz / nist.LIGHT_SPEED_SI * 1e-2
    _print_list("w projection of trans & rot", freq_cm1)
    return freq_cm1

def collect_freq(mass, atom_coords, results, freqs_tr_cm1):
    natm = atom_coords.shape[0]
    # TR 本数
    rot_const  = rotation_const(mass, atom_coords)
    rotor_type = _get_rotor_type(rot_const)
    nTR = 3 if rotor_type=='ATOM' else 5 if rotor_type=='LINEAR' else 6

    # TR (3/5/6)
    tr = numpy.asarray(freqs_tr_cm1).reshape(-1)[:nTR]

    # vib / full 判定
    arr = numpy.asarray(results['freq_wavenumber']).reshape(-1)
    expected_vib = 3*natm - nTR
    if arr.size == expected_vib:
        vib = arr
    elif arr.size == 3*natm:
        vib = arr[-expected_vib:]
    else:
        raise ValueError(f"len(results['freq_wavenumber'])={arr.size} (expected {expected_vib} or {3*natm})")

    full = numpy.concatenate([tr, vib])
    return tr, vib, full, nTR

def mode_type(i, ntr):
    if i < 3:     return "quasi-trans"
    elif i < ntr: return "quasi-rot"
    else:         return "vib"

def print_table_tabulate(freq_all_cm1, ntr):
    rows = [[i+1, mode_type(i, ntr), float(numpy.real(v))]
            for i, v in enumerate(freq_all_cm1)]
    print("Final Result")
    print(tabulate(
        rows,
        headers=["Mode", "type", "vibrational frequency (cm^-1)"],
        tablefmt="psql",
        colalign=("right", "left", "decimal"),
        floatfmt=(".0f", "", ".4f"),
    ))

def show_frequencies(mass, coords, hess, results):
    freq_wo = results['freq_wavenumber']
    freqs_tr = compute_tr_frequencies(hess, mass, coords)
    tr, vib, full, nTR = collect_freq(mass, coords, results, freqs_tr)
    print_table_tabulate(full, ntr=nTR)

    return {
        "freq_wavenumber_wo": freq_wo,   # 3N
        "freqs_tr": freqs_tr,            # 3/5/6
        "freq_vib": vib,                 # 3N-nTR
        "freq_all": full,                # 3N
        "nTR": nTR,
    }

def _S_one(v_cm1, T):
    x = 100.0 * h * c / (k * T)
    t = x * v_cm1
    if t < 1e-12:
        return 0.0
    e = math.exp(-t)
    return R * (t * e / (1.0 - e) - math.log(1.0 - e))

def _E_one(v_cm1, T):
    x = 100.0 * h * c / (k * T)
    t = x * v_cm1
    if t < 1e-12:
        return 0.0
    e = math.exp(-t)
    return R * T * (t * (0.5 + e/(1.0 - e)))

def calc_hsm(tr, vib, T=298.15):
    if tr.size < 3:
        raise ValueError(f"It needs three frequencies to calculate thermodynamics.")
    freq_trans = tr[:3]
    freq_rot   = tr[3:]
    freq_vib   = numpy.real_if_close(numpy.asarray(vib), tol=1e8).astype(float, copy=False)
    
    # calc. entropy [J/mol/K]
    S_trans = sum(_S_one(v, T) for v in freq_trans)
    S_rot   = sum(_S_one(v, T) for v in freq_rot)
    S_vib   = sum(_S_one(v, T) for v in freq_vib)
    S_total = S_trans + S_rot + S_vib

    # calc, internal energy [J/mol]
    E_trans = 0.5 * sum(_E_one(v, T) for v in freq_trans)
    E_rot   = 0.5 * sum(_E_one(v, T) for v in freq_rot)
    E_vib   =       sum(_E_one(v, T) for v in freq_vib)
    E_total = E_trans + E_rot + E_vib

    freq_all = numpy.concatenate([freq_trans, freq_rot, freq_vib])

    return {
        "T": T,
        "freq": {
            "trans": freq_trans, "rot": freq_rot, "vib": freq_vib, "all": freq_all
        },
        "S": {
            "trans": S_trans, "rot": S_rot, "vib": S_vib, "total": S_total
        },
        "E": {
            "trans": E_trans, "rot": E_rot, "vib": E_vib, "total": E_total
        },
        }

def print_hsm_tables(hsm, *, freq_decimals=2, thermo_fmt=".4e"):
    """
    hsm : calc_hsm(...) の戻り値
    """
    T = hsm["T"]
    f  = hsm["freq"]
    S  = hsm["S"]
    E  = hsm["E"]
    kcal = 0.000239006

    rows_thermo = [
        ["E (kJ/mol)",      E["trans"]/1000, E["rot"]/1000, E["vib"]/1000, E["total"]/1000],
        ["S (kJ/mol/K)",    S["trans"]/1000, S["rot"]/1000, S["vib"]/1000, S["total"]/1000],
    ]

    print(f"\nThermochemistry  [T = {T:.2f} K]")
    print(tabulate(
        rows_thermo,
        headers=["Quantity", "Trans", "Rot", "Vib", "Total"],
        tablefmt="psql",
        colalign=("left", "decimal", "decimal", "decimal", "decimal"),
        floatfmt=("", ".5f", ".5f", ".5f", ".5f"),
    ))
    print()
    print("J/mol -> kcal/mol")
    rows_thermo = [
        ["E (kcal/mol)",      E["trans"]*kcal, E["rot"]*kcal, E["vib"]*kcal, E["total"]*kcal],
        ["S (kcal/mol/K)",    S["trans"]*kcal, S["rot"]*kcal, S["vib"]*kcal, S["total"]*kcal],
    ]
    print(tabulate(
        rows_thermo,
        headers=["Quantity", "Trans", "Rot", "Vib", "Total"],
        tablefmt="psql",
        colalign=("left", "decimal", "decimal", "decimal", "decimal"),
        floatfmt=("", ".3f", ".3f", ".3f", ".3f"),
    ))

def rotation_const(mass, atom_coords, unit='GHz'):
    '''Rotational constants to characterize rotational spectra

    Kwargs:
        unit (string) : One of GHz, wavenumber
    '''
    mass_center = numpy.einsum('z,zr->r', mass, atom_coords) / mass.sum()
    r = atom_coords - mass_center
    im = numpy.einsum('z,zr,zs->rs', mass, r, r)
    im = numpy.eye(3) * im.trace() - im
    e = numpy.sort(numpy.linalg.eigvalsh(im))

    unit_im = nist.ATOMIC_MASS * (nist.BOHR_SI)**2
    unit_hz = nist.HBAR / (4 * numpy.pi * unit_im)
    with numpy.errstate(divide='ignore'):
        if unit.lower() == 'ghz':
            e = unit_hz / e * 1e-9
        elif unit.lower() == 'wavenumber':
            e = unit_hz / e / nist.LIGHT_SPEED_SI * 1e-2
        else:
            raise RuntimeError('Unsupported unit ' + unit)
    return e


def thermo(model, freq, temperature=298.15, pressure=101325):
    mol = model.mol
    atom_coords = mol.atom_coords()
    mass = mol.atom_mass_list(isotope_avg=True)
    mass_center = numpy.einsum('z,zx->x', mass, atom_coords) / mass.sum()
    atom_coords = atom_coords - mass_center

    kB = nist.BOLTZMANN
    h = nist.PLANCK
    # c = nist.LIGHT_SPEED_SI
    # beta = 1. / (kB * temperature)
    R_Eh = kB*nist.AVOGADRO / (nist.HARTREE2J * nist.AVOGADRO)

    results = {}
    results['temperature'] = (temperature, 'K')
    results['pressure'] = (pressure, 'Pa')

    E0 = model.e_tot
    results['E0'] = (E0, 'Eh')

    # Electronic part
    results['S_elec' ] = (R_Eh * numpy.log(mol.multiplicity), 'Eh/K')
    results['Cv_elec'] = results['Cp_elec'] = (0, 'Eh/K')
    results['E_elec' ] = results['H_elec' ] = (E0, 'Eh')

    # Translational part. See also https://cccbdb.nist.gov/thermo.asp for the
    # partition function q_trans
    mass_tot = mass.sum() * nist.ATOMIC_MASS
    q_trans = ((2.0 * numpy.pi * mass_tot * kB * temperature / h**2)**1.5
               * kB * temperature / pressure)
    results['S_trans' ] = (R_Eh * (2.5 + numpy.log(q_trans)), 'Eh/K')
    results['Cv_trans'] = (1.5 * R_Eh, 'Eh/K')
    results['Cp_trans'] = (2.5 * R_Eh, 'Eh/K')
    results['E_trans' ] = (1.5 * R_Eh * temperature, 'Eh')
    results['H_trans' ] = (2.5 * R_Eh * temperature, 'Eh')

    # Rotational part
    rot_const = rotation_const(mass, atom_coords, 'GHz')
    results['rot_const'] = (rot_const, 'GHz')
    rotor_type = _get_rotor_type(rot_const)

    sym_number = rotational_symmetry_number(mol)
    results['sym_number'] = (sym_number, '')

    # partition function q_rot (https://cccbdb.nist.gov/thermo.asp)
    if rotor_type == 'ATOM':
        results['S_rot' ] = (0, 'Eh/K')
        results['Cv_rot'] = results['Cp_rot'] = (0, 'Eh/K')
        results['E_rot' ] = results['H_rot' ] = (0, 'Eh')
    elif rotor_type == 'LINEAR':
        B = rot_const[1] * 1e9
        q_rot = kB * temperature / (sym_number * h * B)
        results['S_rot' ] = (R_Eh * (1 + numpy.log(q_rot)), 'Eh/K')
        results['Cv_rot'] = results['Cp_rot'] = (R_Eh, 'Eh/K')
        results['E_rot' ] = results['H_rot' ] = (R_Eh * temperature, 'Eh')
    else:
        ABC = rot_const * 1e9
        q_rot = ((kB*temperature/h)**1.5 * numpy.pi**.5
                 / (sym_number * numpy.prod(ABC)**.5))
        results['S_rot' ] = (R_Eh * (1.5 + numpy.log(q_rot)), 'Eh/K')
        results['Cv_rot'] = results['Cp_rot'] = (1.5 * R_Eh, 'Eh/K')
        results['E_rot' ] = results['H_rot' ] = (1.5 * R_Eh * temperature, 'Eh')

    # Vibrational part.
    au2hz = (nist.HARTREE2J / (nist.ATOMIC_MASS * nist.BOHR_SI**2))**.5 / (2 * numpy.pi)
    idx = freq.real > 0
    vib_temperature = freq.real[idx] * au2hz * h / kB
    # reduced_temperature
    rt = vib_temperature / max(1e-14, temperature)
    e = numpy.exp(-rt)

    ZPE = R_Eh * .5 * vib_temperature.sum()
    results['ZPE'] = (ZPE, 'Eh')

    results['S_vib' ] = (R_Eh * (rt*e/(1-e) - numpy.log(1-e)).sum(), 'Eh/K')
    results['Cv_vib'] = results['Cp_vib'] = (R_Eh * (e * rt**2/(1-e)**2).sum(), 'Eh/K')
    results['E_vib' ] = results['H_vib' ] = \
            (ZPE + R_Eh * temperature * (rt * e / (1-e)).sum(), 'Eh')

    results['G_elec' ] = (results['H_elec' ][0] - temperature * results['S_elec' ][0], 'Eh')
    results['G_trans'] = (results['H_trans'][0] - temperature * results['S_trans'][0], 'Eh')
    results['G_rot'  ] = (results['H_rot'  ][0] - temperature * results['S_rot'  ][0], 'Eh')
    results['G_vib'  ] = (results['H_vib'  ][0] - temperature * results['S_vib'  ][0], 'Eh')

    def _sum(f):
        keys = ('elec', 'trans', 'rot', 'vib')
        return sum(results.get(f+'_'+key, (0,))[0] for key in keys)
    results['S_tot' ] = (_sum('S' ), 'Eh/K')
    results['Cv_tot'] = (_sum('Cv'), 'Eh/K')
    results['Cp_tot'] = (_sum('Cp'), 'Eh/K')
    results['E_0K' ]  = (E0 + ZPE, 'Eh')
    results['E_tot' ] = (_sum('E'), 'Eh')
    results['H_tot' ] = (_sum('H'), 'Eh')
    results['G_tot' ] = (_sum('G'), 'Eh')

    return results


def _get_TR(mass, coords):
    '''Translational mode and rotational mode'''
    mass_center = numpy.einsum('z,zx->x', mass, coords) / mass.sum()
    coords = coords - mass_center
    massp = mass ** .5

    # translational mode
    Tx = numpy.einsum('m,x->mx', massp, [1, 0, 0])
    Ty = numpy.einsum('m,x->mx', massp, [0, 1, 0])
    Tz = numpy.einsum('m,x->mx', massp, [0, 0, 1])

    im = numpy.einsum('m,mx,my->xy', mass, coords, coords)
    im = numpy.eye(3) * im.trace() - im
    w, paxes = numpy.linalg.eigh(im)
    # make the z-axis be the rotation vector with the smallest moment of inertia
    w = w[::-1]
    paxes = paxes[:,::-1]
    ex, ey, ez = paxes.T

    # rotational mode
    coords_in_rot_frame = coords.dot(paxes)
    cx, cy, cz = coords_in_rot_frame.T
    Rx = massp[:,None] * (cy[:,None] * ez - cz[:,None] * ey)
    Ry = massp[:,None] * (cz[:,None] * ex - cx[:,None] * ez)
    Rz = massp[:,None] * (cx[:,None] * ey - cy[:,None] * ex)

    return (Tx.ravel(), Ty.ravel(), Tz.ravel(),
            Rx.ravel(), Ry.ravel(), Rz.ravel())


def _get_rotor_type(rot_const):
    if numpy.all(rot_const > 1e8):
        rotor_type = 'ATOM'
    elif rot_const[0] > 1e8 and (rot_const[1] - rot_const[2] < 1e-3):
        rotor_type = 'LINEAR'
    else:
        rotor_type = 'REGULAR'
    return rotor_type


def rotational_symmetry_number(mol):
    '''Number of unique orientations of the rigid molecule that only
    interchange identical atoms.

    Source http://cccbdb.nist.gov/thermo.asp (search "symmetry number")
    '''
    from pyscf import symm
    group = symm.detect_symm(mol._atom)[0]

    if group in ['SO3', 'C1', 'Ci', 'Cs', 'Coov']:
        sigma = 1
    elif group == 'Dooh':
        sigma = 2
    elif group in ['T', 'Td']:
        sigma = 12
    elif group == 'Oh':
        sigma = 24
    elif group == 'Ih':
        sigma = 60
    elif group[0] == 'C': # 'Cn', 'Cnv', 'Cnh'
        sigma = int(''.join([x for x in group if x.isdigit()]))
    elif group[0] == 'D': # 'Dn', 'Dnd', 'Dnh'
        sigma = 2 * int(''.join([x for x in group if x.isdigit()]))
    elif group[0] == 'S': # 'Sn'
        sigma = int(''.join([x for x in group if x.isdigit()])) / 2
    else:
        raise RuntimeError("symmetry group: " + group)
    return sigma



def dump_thermo(mol, results):
    dump = mol.stdout.write
    dump('temperature %.4f [%s]\n' % results['temperature'])
    dump('pressure %.2f [%s]\n' % results['pressure'])
    dump('Rotational constants [%s] %.5f %.5f %.5f\n'
         % ((results['rot_const'][1],) + tuple(results['rot_const'][0])))
    dump('Symmetry number %d\n' % results['sym_number'][0])
    dump('Zero-point energy (ZPE) %.5f [Eh]   %.3f [J/mol]\n'
         % (results['ZPE'][0], results['ZPE'][0] * nist.HARTREE2J * nist.AVOGADRO))

    keys = ('tot', 'elec', 'trans', 'rot', 'vib')
    dump('                    %s\n' % ' '.join('%10s'%x for x in keys))
    def convert(f, keys, unit):
        if 'Eh' in unit:
            conv = nist.HARTREE2J * nist.AVOGADRO
        else:
            conv = 1
        return ' '.join('%10.3f'%(results.get(f+'_'+key, (0,))[0]*conv) for key in keys)
    def write(title, f):
        tot, unit = results[f+'_tot']
        msg = convert(f, keys, unit)
        unit = unit.replace('Eh', 'J/mol')
        s = '%s [%s]' % (title, unit)
        dump('%-20s %s\n' % (s, msg))
    write('Entropy', 'S')
    write('Cv', 'Cv')
    write('Cp', 'Cp')

    dump('%-28s               %s\n'
         % ('Internal energy [J/mol]', convert('E', keys[2:], 'Eh')))
    dump('%-22s %.5f  %.5f\n'
         % ('Internal energy [Eh]', results['E_tot'][0], results['E0'][0]))
    dump('%-28s               %s\n'
         % ('Enthalpy [J/mol]', convert('H', keys[2:], 'Eh')))
    dump('%-22s %.5f\n'
         % ('Enthalpy [Eh]', results['H_tot'][0]))
    dump('%-28s               %s\n'
         % ('Gibbs free energy [J/mol]', convert('G', keys[2:], 'Eh')))
    dump('%-22s %.5f\n'
         % ('Gibbs free energy [Eh]', results['G_tot'][0]))

def dump_normal_mode(mol, results):
    dump = mol.stdout.write
    freq_wn = results['freq_wavenumber']
    idx = freq_wn.real > 0
    freq_wn = freq_wn.real[idx]
    nfreq = freq_wn.size

    r_mass = results['reduced_mass'].real[idx]
    force = results['force_const_dyne'].real[idx]
    vib_t = results['vib_temperature'].real[idx]
    mode = results['norm_mode'].real[idx]
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]

    def inline(q, col0, col1):
        return ''.join('%20.4f' % q[i] for i in range(col0, col1))
    def mode_inline(row, col0, col1):
        return '  '.join('%6.2f%6.2f%6.2f' % (mode[i,row,0], mode[i,row,1], mode[i,row,2])
                         for i in range(col0, col1))

    for col0, col1 in lib.prange(0, nfreq, 3):
        dump('Mode              %s\n' % ''.join('%20d'%i for i in range(col0,col1)))
        dump('Irrep\n')
        dump('Freq [cm^-1]          %s\n' % inline(freq_wn, col0, col1))
        dump('Reduced mass [au]     %s\n' % inline(r_mass, col0, col1))
        dump('Force const [Dyne/A]  %s\n' % inline(force, col0, col1))
        dump('Char temp [K]         %s\n' % inline(vib_t, col0, col1))
        #dump('IR\n')
        #dump('Raman\n')
        dump('Normal mode            %s\n' % ('       x     y     z'*(col1-col0)))
        for j, at in enumerate(symbols):
            dump('    %4d%4s               %s\n' % (j, at, mode_inline(j, col0, col1)))


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import hessian
    mol = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587')

    mass = mol.atom_mass_list(isotope_avg=True)
    r = mol.atom_coords() - numpy.random.random((1,3))
    print(rotation_const(mass, r, 'GHz'))
    print(rotation_const(mass[1:], r[1:], 'GHz'))
    print(rotation_const(mass[2:], r[2:], 'GHz'))

    mf = mol.apply('HF').run()
    hess = hessian.RHF(mf).kernel()
    results = harmonic_analysis(mol, hess)
    dump_normal_mode(mol, results)

    results = thermo(mf, results['freq_au'], 298.15, 101325)
    dump_thermo(mol, results)


