__all__ = ['ed_1v1c_py_full_double','rixs_1v1c_py_double','rixs_1v1c_py_double_all']

import numpy as np
#import jax
#import jax.numpy as jnp
import scipy
import edrixs_python as edrixs
import sys
sys.modules['edrixs'] = edrixs

import time

from edrixs_python.utils import info_atomic_shell, slater_integrals_name
from edrixs_python.coulomb_utensor import get_umat_slater
from edrixs_python.soc import atom_hsoc
from edrixs_python.angular_momentum import (
    get_sx, get_sy, get_sz, get_lx, get_ly, get_lz)
from edrixs_python.fock_basis import get_fock_bin_by_N
from edrixs_python.manybody_operator import two_fermion, four_fermion
from edrixs_python.photon_transition import (
    get_trans_oper,dipole_polvec_rixs)
from edrixs_python.rixs_utils import scattering_mat

def saveops(shell_name,v_noccu,*,on_which='spin',loc_axis=None):
    v_name = shell_name[0].strip()
    c_name = shell_name[1].strip()
    slater_name = slater_integrals_name((v_name, c_name), ('v', 'c'))
    nslat = len(slater_name)

    case = v_name + c_name

    info_shell = info_atomic_shell()

    # Quantum numbers of angular momentum
    v_orbl = info_shell[v_name][0]

    v_norb = info_shell[v_name][1]
    c_norb = info_shell[c_name][1]
    ntot = v_norb + c_norb

    basis_i = get_fock_bin_by_N(v_norb, v_noccu, c_norb, c_norb)
    basis_n = get_fock_bin_by_N(v_norb, v_noccu+1, c_norb, c_norb - 1)
    ncfg_i, ncfg_n = len(basis_i), len(basis_n)

    sptab2=[]
    emat_i = np.zeros((ntot, ntot), dtype=complex)
    emat_n = np.zeros((ntot, ntot), dtype=complex)
    emat_i[0:v_norb, 0:v_norb] = atom_hsoc(v_name, 1.0)
    emat_n[0:v_norb, 0:v_norb] += atom_hsoc(v_name, 1.0)
    sptab2.append(scipy.sparse.csr_array(two_fermion(emat_i, basis_i, basis_i))) # 0
    sptab2.append(scipy.sparse.csr_array(two_fermion(emat_n, basis_n, basis_n))) # 1

    emat_n = np.zeros((ntot, ntot), dtype=complex)
    if c_name != 'p32':
       emat_n[v_norb:ntot, v_norb:ntot] += atom_hsoc(c_name, 1.0)
    sptab2.append(scipy.sparse.csr_array(two_fermion(emat_n, basis_n, basis_n))) # 2

    v_cfmat = edrixs.cf_cubic_d(1.0)

    emat_i = np.zeros((ntot, ntot), dtype=complex)
    emat_n = np.zeros((ntot, ntot), dtype=complex)
    emat_i[0:v_norb, 0:v_norb] += np.array(v_cfmat)
    emat_n[0:v_norb, 0:v_norb] += np.array(v_cfmat)
    sptab2.append(scipy.sparse.csr_array(two_fermion(emat_i, basis_i, basis_i))) # 3
    sptab2.append(scipy.sparse.csr_array(two_fermion(emat_n, basis_n, basis_n))) # 4

    emat_i = np.zeros((ntot, ntot), dtype=complex)
    emat_n = np.zeros((ntot, ntot), dtype=complex)
    emat_i[0:v_norb, 0:v_norb] = np.eye(v_norb) * 1.0
    emat_n[0:v_norb, 0:v_norb] = np.eye(v_norb) * 1.0
    sptab2.append(scipy.sparse.csr_array(two_fermion(emat_i, basis_i, basis_i))) # 5
    sptab2.append(scipy.sparse.csr_array(two_fermion(emat_n, basis_n, basis_n))) # 6

    emat_i = np.zeros((ntot, ntot), dtype=complex)
    emat_n = np.zeros((ntot, ntot), dtype=complex)
    emat_i[v_norb:ntot, v_norb:ntot] = np.eye(c_norb) * 1.0
    emat_n[v_norb:ntot, v_norb:ntot] = np.eye(c_norb) * 1.0
    sptab2.append(scipy.sparse.csr_array(two_fermion(emat_i, basis_i, basis_i))) # 7
    sptab2.append(scipy.sparse.csr_array(two_fermion(emat_n, basis_n, basis_n))) # 8

    # external magnetic field
    if v_name == 't2g':
        lx, ly, lz = get_lx(1, True), get_ly(1, True), get_lz(1, True)
        sx, sy, sz = get_sx(1), get_sy(1), get_sz(1)
        lx, ly, lz = -lx, -ly, -lz
    else:
        lx, ly, lz = get_lx(v_orbl, True), get_ly(v_orbl, True), get_lz(v_orbl, True)
        sx, sy, sz = get_sx(v_orbl), get_sy(v_orbl), get_sz(v_orbl)

    emat_i = np.zeros((ntot, ntot), dtype=complex)
    emat_n = np.zeros((ntot, ntot), dtype=complex)
    if on_which.strip() == 'spin':
        b0 = (2 * sx)
        b1 = (2 * sy)
        b2 = (2 * sz)
    elif on_which.strip() == 'orbital':
        b0 = lx
        b1 = ly
        b2 = lz
    elif on_which.strip() == 'both':
        b0 = (lx + 2 * sx)
        b1 = (ly + 2 * sy)
        b2 = (lz + 2 * sz)
    else:
        raise Exception("Unknown value of on_which", on_which)
    emat_i[0:v_norb, 0:v_norb] = b0
    emat_n[0:v_norb, 0:v_norb] = b0
    sptab2.append(scipy.sparse.csr_array(two_fermion(emat_i, basis_i, basis_i))) # 9
    sptab2.append(scipy.sparse.csr_array(two_fermion(emat_n, basis_n, basis_n))) # 10
    emat_i = np.zeros((ntot, ntot), dtype=complex)
    emat_n = np.zeros((ntot, ntot), dtype=complex)
    emat_i[0:v_norb, 0:v_norb] = b1
    emat_n[0:v_norb, 0:v_norb] = b1
    sptab2.append(scipy.sparse.csr_array(two_fermion(emat_i, basis_i, basis_i))) # 11
    sptab2.append(scipy.sparse.csr_array(two_fermion(emat_n, basis_n, basis_n))) # 12
    emat_i = np.zeros((ntot, ntot), dtype=complex)
    emat_n = np.zeros((ntot, ntot), dtype=complex)
    emat_i[0:v_norb, 0:v_norb] = b2
    emat_n[0:v_norb, 0:v_norb] = b2
    sptab2.append(scipy.sparse.csr_array(two_fermion(emat_i, basis_i, basis_i))) # 13
    sptab2.append(scipy.sparse.csr_array(two_fermion(emat_n, basis_n, basis_n))) # 14

    sptab4=[]
    for which in range(nslat):
        slater_i = np.zeros(nslat, dtype=float)
        slater_i[which]=1.0
        umat_i = get_umat_slater(case, *slater_i)
        sptab4.append(scipy.sparse.csr_array(four_fermion(umat_i, basis_i)))
        sptab4.append(scipy.sparse.csr_array(four_fermion(umat_i, basis_n)))

    sptabtr=[]
    if loc_axis is not None:
        local_axis = np.array(loc_axis)
    else:
        local_axis = np.eye(3)
    tmp = get_trans_oper(case)
    npol, n, m = tmp.shape
    tmp_g = np.zeros((npol, n, m), dtype=complex)

    # Transform the transition operators to global-xyz axis
    # dipolar transition
    if npol == 3:
        for i in range(3):
            for j in range(3):
                tmp_g[i] += local_axis[i, j] * tmp[j]

    # quadrupolar transition
    elif npol == 5:
        alpha, beta, gamma = rmat_to_euler(local_axis)
        wignerD = get_wigner_dmat(4, alpha, beta, gamma)
        rotmat = np.dot(np.dot(tmat_r2c('d'), wignerD), np.conj(np.transpose(tmat_r2c('d'))))
        for i in range(5):
            for j in range(5):
                tmp_g[i] += rotmat[i, j] * tmp[j]
    else:
        raise Exception("Have NOT implemented this case: ", npol)

    tmp2 = np.zeros((npol, ntot, ntot), dtype=complex)
    trans_op = np.zeros((npol, ncfg_n, ncfg_i), dtype=complex)
    for i in range(npol):
        tmp2[i, 0:v_norb, v_norb:ntot] = tmp_g[i]
        sptabtr.append(scipy.sparse.csr_array(two_fermion(tmp2[i], basis_n, basis_i)))

    return sptab2, sptab4, sptabtr

def ed_1v1c_init(shell_name, sptab2, sptab4,*, shell_level=None, v_soc=None, c_soc=0,
               v_noccu=1, slater=None, ext_B=None,
               tenDq=0 ):

    print("edrixs >>> Running ED ...")
    v_name_options = ['s', 'p', 't2g', 'd', 'f']
    c_name_options = ['s', 'p', 'p12', 'p32', 't2g', 'd', 'd32', 'd52', 'f', 'f52', 'f72']
    v_name = shell_name[0].strip()
    c_name = shell_name[1].strip()
    if v_name not in v_name_options:
        raise Exception("NOT supported type of valence shell: ", v_name)
    if c_name not in c_name_options:
        raise Exception("NOT supported type of core shell: ", c_name)

    info_shell = info_atomic_shell()

    # Quantum numbers of angular momentum
    v_orbl = info_shell[v_name][0]

    # number of orbitals including spin degree of freedom
    v_norb = info_shell[v_name][1]
    c_norb = info_shell[c_name][1]

    # total number of orbitals
    ntot = v_norb + c_norb

    # Coulomb interaction
    # Get the names of all the required slater integrals
    slater_name = slater_integrals_name((v_name, c_name), ('v', 'c'))
    nslat = len(slater_name)

    slater_i = np.zeros(nslat, dtype=float)

    if slater is not None:
        if nslat > len(slater[0]):
            slater_i[0:len(slater[0])] = slater[0]
        else:
            slater_i[:] = slater[0][0:nslat]

    case = v_name + c_name

    Hmat_isp=scipy.sparse.csr_array(sptab4[0].shape,dtype=np.complex128)

    # SOC
    if v_soc is not None:
        Hmat_isp += v_soc[0]*sptab2[0]

    # crystal field
    Hmat_isp += tenDq*sptab2[3]

    # energy of shells
    if shell_level is not None:
        Hmat_isp += shell_level[0]*sptab2[5]
        Hmat_isp += shell_level[1]*sptab2[7]

    # external magnetic field

    if ext_B is not None:
        Hmat_isp += ext_B[0]*sptab2[9]
        Hmat_isp += ext_B[1]*sptab2[11]
        Hmat_isp += ext_B[2]*sptab2[13]

    # Build many-body Hamiltonian in Fock basis
    print("edrixs >>> Building Many-body Hamiltonians ...")
    for i in range(nslat):
        Hmat_isp += slater_i[i]*sptab4[2*i]

    hmat_i = Hmat_isp.toarray()

    #print("edrixs >>> Done !")
    # Do exact-diagonalization to get eigenvalues and eigenvectors
    print("edrixs >>> Exact Diagonalization of Hamiltonians ...")
    eval_i, evec_i = scipy.linalg.eigh(hmat_i)

    print("edrixs >>> Done !")

    return eval_i, evec_i

def ed_1v1c_py_full_double(shell_name, sptab2, sptab4, sptabtr,*, shell_level=None, v_soc=None, c_soc=0,
               v_noccu=1, slater=None, ext_B=None,
               tenDq=0 ):

    print("edrixs >>> Running ED ...")
    v_name_options = ['s', 'p', 't2g', 'd', 'f']
    c_name_options = ['s', 'p', 'p12', 'p32', 't2g', 'd', 'd32', 'd52', 'f', 'f52', 'f72']
    v_name = shell_name[0].strip()
    c_name = shell_name[1].strip()
    if v_name not in v_name_options:
        raise Exception("NOT supported type of valence shell: ", v_name)
    if c_name not in c_name_options:
        raise Exception("NOT supported type of core shell: ", c_name)

    info_shell = info_atomic_shell()

    # Quantum numbers of angular momentum
    v_orbl = info_shell[v_name][0]

    # number of orbitals including spin degree of freedom
    v_norb = info_shell[v_name][1]
    c_norb = info_shell[c_name][1]

    # total number of orbitals
    ntot = v_norb + c_norb

    # Coulomb interaction
    # Get the names of all the required slater integrals
    slater_name = slater_integrals_name((v_name, c_name), ('v', 'c'))
    nslat = len(slater_name)

    slater_i = np.zeros(nslat, dtype=float)
    slater_n = np.zeros(nslat, dtype=float)

    if slater is not None:
        if nslat > len(slater[0]):
            slater_i[0:len(slater[0])] = slater[0]
        else:
            slater_i[:] = slater[0][0:nslat]
        if nslat > len(slater[1]):
            slater_n[0:len(slater[1])] = slater[1]
        else:
            slater_n[:] = slater[1][0:nslat]

    # print summary of slater integrals
    print()
    print("    Summary of Slater integrals:")
    print("    ------------------------------")
    print("    Terms,   Initial Hamiltonian,  Intermediate Hamiltonian")
    for i in range(nslat):
        print("    ", slater_name[i], ":  {:20.10f}{:20.10f}".format(slater_i[i], slater_n[i]))
    print()

    case = v_name + c_name

    Hmat_isp=scipy.sparse.csr_array(sptab4[0].shape,dtype=np.complex128)
    Hmat_nsp=scipy.sparse.csr_array(sptab4[1].shape,dtype=np.complex128)
#    umat_i = get_umat_slater(case, *slater_i)
#    umat_n = get_umat_slater(case, *slater_n)

    # SOC
    if v_soc is not None:
        Hmat_isp += v_soc[0]*sptab2[0]
        Hmat_nsp += v_soc[1]*sptab2[1]

    # when the core-shell is any of p12, p32, d32, d52, f52, f72,
    # do not need to add SOC for core shell
    if c_name in ['p', 'd', 'f']:
        Hmat_nsp += c_soc*sptab2[2]

    # crystal field
    Hmat_isp += tenDq*sptab2[3]
    Hmat_nsp += tenDq*sptab2[4]

    # other hopping matrix
#    if v_othermat is not None:
#        emat_i[0:v_norb, 0:v_norb] += np.array(v_othermat)
#        emat_n[0:v_norb, 0:v_norb] += np.array(v_othermat)

    # energy of shells
    if shell_level is not None:
        Hmat_isp += shell_level[0]*sptab2[5]
        Hmat_nsp += shell_level[0]*sptab2[6]
        Hmat_isp += shell_level[1]*sptab2[7]
        Hmat_nsp += shell_level[1]*sptab2[8]

    # external magnetic field

    if ext_B is not None:
        Hmat_isp += ext_B[0]*sptab2[9]
        Hmat_nsp += ext_B[0]*sptab2[10]
        Hmat_isp += ext_B[1]*sptab2[11]
        Hmat_nsp += ext_B[1]*sptab2[12]
        Hmat_isp += ext_B[2]*sptab2[13]
        Hmat_nsp += ext_B[2]*sptab2[14]

    # Build many-body Hamiltonian in Fock basis
    print("edrixs >>> Building Many-body Hamiltonians ...")
    for i in range(nslat):
        Hmat_isp += slater_i[i]*sptab4[2*i]
        Hmat_nsp += slater_n[i]*sptab4[2*i+1]

    hmat_i = Hmat_isp.toarray()
    hmat_n = Hmat_nsp.toarray()

    #print("edrixs >>> Done !")

    # Do exact-diagonalization to get eigenvalues and eigenvectors
    print("edrixs >>> Exact Diagonalization of Hamiltonians ...")
    eval_i, evec_i = scipy.linalg.eigh(hmat_i)

    hmat_n=hmat_n.astype(np.complex128)

#    eval_n, evec_n = scipy.linalg.eigh(hmat_n)
    eval_n, evec_n = scipy.linalg.eigh(hmat_n)
    print("edrixs >>> Done !")
#    t=time.time()
#    invm = scipy.linalg.inv(hmat_n)
#    t=time.time()-t
#    print("Time spent with inversion: "+str(t))

        # Build dipolar transition operators in local-xyz axis
    tmp = get_trans_oper(case)
    npol, n, m = tmp.shape

    trans_op = np.zeros((npol, sptabtr[0].shape[0], sptabtr[0].shape[1]), dtype=complex)
    for i in range(npol):
        trans_op[i] = cb_op2(sptabtr[i].toarray(), evec_n, evec_i)
    print("edrixs >>> ED Done !")

    return eval_i, eval_n, trans_op, evec_i, evec_n


def cb_op2(oper_O, TL, TR):
    oper_O = np.array(oper_O, order='K')
    dim = oper_O.shape
    if len(dim) < 2:
        raise Exception("Dimension of oper_O should be at least 2")
    elif len(dim) == 2:
        res = np.dot(np.dot(np.conj(np.transpose(TL)), oper_O), TR)
    else:
        tot = np.prod(dim[0:-2])
        tmp_oper = oper_O.reshape((tot, dim[-2], dim[-1]))
        for i in range(tot):
            tmp_oper[i] = np.dot(np.dot(np.conj(np.transpose(TL)), tmp_oper[i]), TR)
        res = tmp_oper.reshape(dim)

    return res

def Ctensor(eval_i, gs_list,eval_n,omega_inc,eloss,gamma_n,gamma_f):
    num_gs = trans_mat_abs.shape[2]
    num_ex = trans_mat_abs.shape[1]
    num_fs = trans_mat_emi.shape[1]

    num_om = omega_inc.shape[0]
    num_el = eloss.shape[0]
    gamma_final = np.zeros(num_el, dtype=float)
    denomi=(np.tensordot(omega_inc,np.ones([num_ex,num_gs]),axes=0) + 1j*gamma_n*np.ones([num_om,num_ex,num_gs])
            - np.tensordot(np.ones(num_om),np.outer(eval_n,np.ones(num_gs)) - np.outer(np.ones(num_ex),eval_i[:num_gs]),axes=0))
    denomi=1/denomi
    rhotab=np.zeros([num_fs,num_gs,num_om,num_el])

    for n in range(0,len(eval_i)):
        for m, igs in enumerate(gs_list):
            rhotab[n,m,:,:]=(np.outer(prob[m] * np.ones(len(omega_inc) , gamma_final / np.pi)) /
                        (np.outer(np.ones(len(omega_inc)),eloss - (eval_i[n] - eval_i[igs]))**2 + np.outer(np.ones(len(omega_inc)),gamma_final)**2))
    
    Rtab2 = np.sum(rhotab,axis=3)

    return np.tensordot(rhotab,denomi,axes=1)

def Dtensor(eval_i, eval_n, trans_mat_abs,
                   trans_mat_emi, Pmat, omega_inc, gamma_n):
    return 0


def scattering_mat2(eval_i, eval_n, trans_mat_abs,
                   trans_mat_emi, omega_inc, gamma_n):

    num_gs = trans_mat_abs.shape[2]
    num_ex = trans_mat_abs.shape[1]
    num_fs = trans_mat_emi.shape[1]

    num_om = omega_inc.shape[0]
#    num_oms=omega_inc_list.shape[0]

    npol_abs = trans_mat_abs.shape[0]
    npol_emi = trans_mat_emi.shape[0]

    Ffi = np.zeros((npol_emi, npol_abs, num_om, num_fs, num_gs), dtype=np.complex128)
    tmp_abs = np.zeros((npol_abs, num_om,num_ex, num_gs), dtype=np.complex128)
    denomi = np.zeros((num_om,num_ex, num_gs), dtype=np.complex128)


#    for i in range(num_ex):
#        for j in range(num_gs):
            #aa = omega_inc - (eval_n[i] - eval_i[j])
#            denomi[i, j] = 1.0 / (aa + 1j * gamma_n)

    denomi=(np.tensordot(omega_inc,np.ones([num_ex,num_gs]),axes=0) + 1j*gamma_n*np.ones([num_om,num_ex,num_gs])
            - np.tensordot(np.ones(num_om),np.outer(eval_n,np.ones(num_gs)) - np.outer(np.ones(num_ex),eval_i[:num_gs]),axes=0))
    denomi=1/denomi


    denomi = np.tensordot(np.ones(npol_abs),denomi,axes=0)
    tmp_abs= np.transpose(np.tensordot(np.ones(num_om),trans_mat_abs,axes=0),axes=[1,0,2,3])*denomi
#    for i in range(npol_abs):
#        tmp_abs[i] = trans_mat_abs[i] * denomi

#    Ffi = np.matmul(np.transpose(np.tensordot(np.ones([npol_abs,num_om]),trans_mat_emi,axes=0),axes=[2,0,1,3,4]),np.tensordot(np.ones(npol_emi),tmp_abs,axes=0))

    for i in range(npol_emi):
        for j in range(npol_abs):
            for k in range(num_om):
                Ffi[i, j, k, :, :] = np.dot(trans_mat_emi[i], tmp_abs[j,k])

    return Ffi

def rixs_1v1c_py_double(eval_i, eval_n, trans_op, ominc, eloss, *,
                 gamma_c=0.1, gamma_f=0.01, thin=1.0, thout=1.0, phi=0.0,
                 pol_type=None, gs_list=None, temperature=1.0, scatter_axis=None, isPowder=False):

    print("edrixs >>> Running RIXS ... ")
    n_ominc = len(ominc)
    n_eloss = len(eloss)
    gamma_core = np.zeros(n_ominc, dtype=float)
    gamma_final = np.zeros(n_eloss, dtype=float)
    if np.isscalar(gamma_c):
        gamma_core[:] = np.ones(n_ominc) * gamma_c
    else:
        gamma_core[:] = gamma_c

    if np.isscalar(gamma_f):
        gamma_final[:] = np.ones(n_eloss) * gamma_f
    else:
        gamma_final[:] = gamma_f

    if pol_type is None:
        pol_type = [('linear', 0, 'linear', 0)]
    if gs_list is None:
        gs_list = [0]
    if scatter_axis is None:
        scatter_axis = np.eye(3)
    else:
        scatter_axis = np.array(scatter_axis)

    prob = edrixs.boltz_dist([eval_i[i] for i in gs_list], temperature)
    rixs = np.zeros((len(ominc), len(eloss), len(pol_type)), dtype=float)
    npol, n, m = trans_op.shape
    trans_emi = np.zeros((npol, m, n), dtype=np.complex128)
    for i in range(npol):
        trans_emi[i] = np.conj(np.transpose(trans_op[i]))
    polvec_i = np.zeros(npol, dtype=complex)
    polvec_f = np.zeros(npol, dtype=complex)

    # Calculate RIXS
#    for i, om in enumerate(ominc):
    t = time.time()
    F_fi = scattering_mat2(eval_i, eval_n, trans_op[:, :, 0:max(gs_list)+1],
                              trans_emi, ominc, gamma_core[i])
    t=time.time()-t
    print("Time spent with scattering_mat2: "+str(t))

    for j, (it, alpha, jt, beta) in enumerate(pol_type):
            ei, ef = dipole_polvec_rixs(thin, thout, phi, alpha, beta,
                                        scatter_axis, (it, jt))
            if isPowder:
               ei=np.ones(3)/np.sqrt(3)                        # modified for o
               ef=np.ones(3)/np.sqrt(3)                        #
            # dipolar transition
            if npol == 3:
                polvec_i[:] = ei
                polvec_f[:] = ef
            # quadrupolar transition
            elif npol == 5:
                ki = unit_wavevector(thin, phi, scatter_axis, direction='in')
                kf = unit_wavevector(thout, phi, scatter_axis, direction='out')
                polvec_i[:] = quadrupole_polvec(ei, ki)
                polvec_f[:] = quadrupole_polvec(ef, kf)
            else:
                raise Exception("Have NOT implemented this type of transition operators")
            # scattering magnitude with polarization vectors
            F_mag = np.zeros((len(ominc), len(eval_i), len(gs_list)), dtype=complex)

            for m in range(npol):
                for n in range(npol):
                    F_mag[:,:, :] += np.conj(polvec_f[m]) * F_fi[m, n] * polvec_i[n]

            for m, igs in enumerate(gs_list):
                for n in range(len(gs_list),len(eval_i)):
                    rixs[:, :, j] += (
                        np.outer(prob[m] * np.abs(F_mag[:,n, igs])**2 , gamma_final / np.pi) /
                        (np.outer(np.ones(len(ominc)),eloss - (eval_i[n] - eval_i[igs]))**2 + np.outer(np.ones(len(ominc)),gamma_final)**2)
                    )
    print("edrixs >>> RIXS Done !")

    return rixs

def rixs_1v1c_py_double_all(eval_i, eval_n, trans_op, ominc, eloss, *,
                 gamma_c=0.1, gamma_f=0.01, thin=1.0, thout=1.0, phi=0.0,
                 pol_type=None, gs_list=None, temperature=1.0, scatter_axis=None, isPowder=False):

    print("edrixs >>> Running RIXS ... ")
    n_ominc = len(ominc)
    n_eloss = len(eloss)
    gamma_core = np.zeros(n_ominc, dtype=float)
    gamma_final = np.zeros(n_eloss, dtype=float)
    if np.isscalar(gamma_c):
        gamma_core[:] = np.ones(n_ominc) * gamma_c
    else:
        gamma_core[:] = gamma_c

    if np.isscalar(gamma_f):
        gamma_final[:] = np.ones(n_eloss) * gamma_f
    else:
        gamma_final[:] = gamma_f

    if pol_type is None:
        pol_type = [('linear', 0, 'linear', 0)]
    if gs_list is None:
        gs_list = [0]
    if scatter_axis is None:
        scatter_axis = np.eye(3)
    else:
        scatter_axis = np.array(scatter_axis)

    prob = edrixs.boltz_dist([eval_i[i] for i in gs_list], temperature)
    rixs = np.zeros((len(ominc), len(eloss), len(pol_type)), dtype=float)
    npol, n, m = trans_op.shape
    trans_emi = np.zeros((npol, m, n), dtype=np.complex128)
    for i in range(npol):
        trans_emi[i] = np.conj(np.transpose(trans_op[i]))
    polvec_i = np.zeros(npol, dtype=complex)
    polvec_f = np.zeros(npol, dtype=complex)

    # Calculate RIXS
#    for i, om in enumerate(ominc):
    t = time.time()
    F_fi = scattering_mat2(eval_i, eval_n, trans_op[:, :, 0:max(gs_list)+1],
                              trans_emi, ominc, gamma_core[i])
    t=time.time()-t
    print("Time spent with scattering_mat2: "+str(t))

    for j, (it, alpha, jt, beta) in enumerate(pol_type):
            ei, ef = dipole_polvec_rixs(thin, thout, phi, alpha, beta,
                                        scatter_axis, (it, jt))
            if isPowder:
               ei=np.ones(3)/np.sqrt(3)                        # modified for o
               ef=np.ones(3)/np.sqrt(3)                        #
            # dipolar transition
            if npol == 3:
                polvec_i[:] = ei
                polvec_f[:] = ef
            # quadrupolar transition
            elif npol == 5:
                ki = unit_wavevector(thin, phi, scatter_axis, direction='in')
                kf = unit_wavevector(thout, phi, scatter_axis, direction='out')
                polvec_i[:] = quadrupole_polvec(ei, ki)
                polvec_f[:] = quadrupole_polvec(ef, kf)
            else:
                raise Exception("Have NOT implemented this type of transition operators")
            # scattering magnitude with polarization vectors
            F_mag = np.zeros((len(ominc), len(eval_i), len(gs_list)), dtype=complex)

            for m in range(npol):
                for n in range(npol):
                    F_mag[:,:, :] += np.conj(polvec_f[m]) * F_fi[m, n] * polvec_i[n]

            for m, igs in enumerate(gs_list):
                for n in range(0,len(eval_i)):
                    rixs[:, :, j] += (
                        np.outer(prob[m] * np.abs(F_mag[:,n, igs])**2 , gamma_final / np.pi) /
                        (np.outer(np.ones(len(ominc)),eloss - (eval_i[n] - eval_i[igs]))**2 + np.outer(np.ones(len(ominc)),gamma_final)**2)
                    )
    print("edrixs >>> RIXS Done !")

    return rixs