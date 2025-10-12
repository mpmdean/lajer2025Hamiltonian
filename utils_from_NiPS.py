""" Various utilities copied from NiPS3_edRIXS project """

import edrixs_python as edrixs
import sys
sys.modules['edrixs'] = edrixs
from edrixs_python import scattering_mat
import numpy as np
import scipy
import ipyparallel as ipp
import contextlib
from solvers_fast_double import ed_1v1c_py_full_double, saveops

import pickle

def get_pol(pol, tth=150., thin=10., phi=0.):
    """ Manually calculate polarization vectors by tranferring from experimental geometry into octahedral coordinates. """
    # for 0KL plane, phi=90
    # for H0L plane, phi=0
    if pol=='LH':
        ei, ef1 = edrixs.dipole_polvec_rixs(thin, tth-thin, phi=phi, alpha=0., beta=0.)
        _,  ef2 = edrixs.dipole_polvec_rixs(thin, tth-thin, phi=phi, alpha=0., beta=np.pi/2)
    elif pol=='LV':
        ei, ef1 = edrixs.dipole_polvec_rixs(thin, tth-thin, phi=phi, alpha=np.pi/2, beta=0.)
        _,  ef2 = edrixs.dipole_polvec_rixs(thin, tth-thin, phi=phi, alpha=np.pi/2, beta=np.pi/2)

    tmat = rmat_t2o(1)
    return np.dot(ei, tmat), np.dot(ef1, tmat), np.dot(ef2, tmat)

def load_var(filename, form='pickle'):
    """
    Load the variable.
    Example
    -------
    >>> data = load_var('data.pkl')
    """
    if form=='numpy':
        data = np.load(filename)
    elif form=='pickle':
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        file.close
    return data

def calc_Slater(U_dd=10., J_dd=1., U_pp=None, J_pp=None):
    """ Calculate Slater integrals with given U and J. """
    F4F2_ratio = 0.625
    F2_dd = J_dd * 14. / (1. + F4F2_ratio)
    F4_dd = F2_dd * F4F2_ratio
    F0_dd = U_dd - 4.0 / 49.0 * F2_dd - 4.0 / 49.0 * F4_dd
    if U_pp is not None and J_pp is not None:
        F2_pp = J_pp * 25 / 3
        F0_pp = U_pp - F2_pp * 4 / 25
    else:
        F0_pp = None
        F2_pp = None
    return F0_dd, F2_dd, F4_dd, F0_pp, F2_pp

def perform_ED_1site(tenDq=0., soc_v_i=0, soc_v_n=0., soc_c=0.,
                     F0_dd=0., F2_dd=0., F4_dd=0.,
                     F0_dp=0., F2_dp=0., G1_dp=0., G3_dp=0.):
    """
    Perform ED calculations for a single transition metal (TM) site atomic model.

    Parameters
    ----------
    tenDq : float
        Crystal field splitting 10Dq=\epsilon(t2g)-\epsilon(eg) in hole language.
    soc_v_i, soc_v_n, soc_c : float
        Spin orbital coupling strength for the initial state d orbitals,
        intermediate state d orbitlas and core orbitals, respectively.
    F0_dd, F2_dd, F4_dd : float
        Slater integrals for TM onsite Coulomb interactions.
    F0_dp, F2_dp, G1_dp, G3_dp : float
        Slater integrals for TM core-hole potential.

    Returns
    -------
    evals_i : 1d array
        Eigen values for the initial states.
    evecs_i : 2d array
        Eigen vectors for the initial states.
    evals_n : 1d array
        Eigen values for the intermediate states.
    evecs_n : 2d array
        Eigen vectors for the intermediate states.
    T_abs : 3d array
        Absorption operator.

    Examples
    --------
    >>> evals_i, evecs_i, evals_n, evecs_n, T_abs = perform_ED_1site(tenDq=1.07)
    """
    # Construct emat
    emat_i = np.zeros((16,16), dtype=complex) # 10 dorb, 6 core
    emat_n = np.zeros((16,16), dtype=complex)
    ## CEF
    emat_tmp = np.diag([0., tenDq, tenDq, 0., tenDq])
    emat_i[:10:2,:10:2]   += emat_tmp
    emat_i[1:10:2,1:10:2] += emat_tmp
    emat_n[:10:2,:10:2]   += emat_tmp
    emat_n[1:10:2,1:10:2] += emat_tmp
    ## SOC
    emat_i_soc = edrixs.cb_op(edrixs.atom_hsoc('d', -soc_v_i), edrixs.tmat_c2r('d', True))
    emat_n_soc = edrixs.cb_op(edrixs.atom_hsoc('d', -soc_v_n), edrixs.tmat_c2r('d', True))
    emat_c_soc = edrixs.cb_op(edrixs.atom_hsoc('p', -soc_c), edrixs.tmat_c2r('p', True))
    emat_i[:10,:10] += emat_i_soc
    emat_n[:10,:10] += emat_n_soc
    emat_n[10:,10:] += emat_c_soc

    # Construct umat
    tmat = scipy.linalg.block_diag(edrixs.tmat_c2r('d',True), edrixs.tmat_c2r('p',True))
    umat_i = edrixs.transform_utensor(edrixs.get_umat_slater('dp', F0_dd, F2_dd, F4_dd, 0, 0, 0, 0, 0., 0.), tmat)
    umat_n = edrixs.transform_utensor(edrixs.get_umat_slater('dp', F0_dd, F2_dd, F4_dd, F0_dp, F2_dp, G1_dp, G3_dp, 0., 0.), tmat)

    # Construct Hartree-Fock basis and Hamiltonian
    basis_i = edrixs.get_fock_bin_by_N(10,2,6,0)  # (10,2,6,0)
    basis_n = edrixs.get_fock_bin_by_N(10,1,6,1)  # (10,1,6,1)
    hmat_i = edrixs.two_fermion(emat_i, basis_i) + edrixs.four_fermion(umat_i, basis_i)
    hmat_n = edrixs.two_fermion(emat_n, basis_n) + edrixs.four_fermion(umat_n, basis_n)
    evals_i, evecs_i = np.linalg.eigh(hmat_i)
    evals_n, evecs_n = np.linalg.eigh(hmat_n)

    # construct transition operator
    ncfg_i, ncfg_n = len(basis_i), len(basis_n)
    dipole = np.zeros((3, 16, 16), dtype=complex)
    T_abs = np.zeros((3, ncfg_n, ncfg_i), dtype=complex)
    tmp = edrixs.cb_op(edrixs.get_trans_oper('dp'), edrixs.tmat_c2r('d', True), edrixs.tmat_c2r('p', True))
    for xyz in range(3):
        dipole[xyz, 10:, :10] = np.transpose(np.conj(tmp[xyz]))
        T_abs[xyz] = edrixs.two_fermion(dipole[xyz], basis_n, basis_i)
        T_abs[xyz] = edrixs.cb_op2(T_abs[xyz], evecs_n, evecs_i)

    return evals_i, evecs_i, evals_n, evecs_n, T_abs

def rmat_t2o(index):
    """
    Get rotational matrix from trigonal to octehedral notations for face-sharing scenario.

    Parameters
    ----------
    index : int, index of transition metal (1 or 2).

    Returns
    -------
    rmat : 3x3 2d arrays, dtype=float.
    """
    if index%2==1:
        rmat = np.zeros((3,3), dtype=float)
        rmat[:,0] = [np.sqrt(1/6), -np.sqrt(1/2), np.sqrt(1/3)]
        rmat[:,1] = [np.sqrt(1/6), np.sqrt(1/2), np.sqrt(1/3)]
        rmat[:,2] = [-np.sqrt(2/3), 0., np.sqrt(1/3)]
    else:
        rmat = np.zeros((3,3), dtype=float)
        rmat[:,0] = [-np.sqrt(1/6), np.sqrt(1/2), np.sqrt(1/3)]
        rmat[:,1] = [-np.sqrt(1/6), -np.sqrt(1/2), np.sqrt(1/3)]
        rmat[:,2] = [np.sqrt(2/3), 0., np.sqrt(1/3)]
    return rmat

def calc_RIXS_Q0(evals_i, evals_n, T_abs,
                 Gam_c=0.6, sigma=0.1, sigmarel=0.0, fraction=0.5,
                 omega=None, eloss=None,
                 phi=0., thin=15., tth=150., pol=[(0,0), (0,np.pi/2)], scatter_axis=None,
                 ei=None, ef=None,
                 T=10., tol=1e-5, ngs=None):
    """
    Calculate the RIXS spectra for Q=0 case.

    Parameters
    ----------
         evals_i : 1d array, eigen values for the initial states.
         evals_n : 1d array, eigen values for the intermediate states.
           T_abs : 2d array, the absorption matrix.
           Gam_c : float, inverse core-hole lifetime in unit of eV (default: 0.6).
           sigma : float, sigma of PseudoVoigt profile (default: 0.1).
        fraction : float, fraction of PseudoVoigt profile (default: 0.5).
           omega : 1d array or float, incident energy. If None, determined by evals (default: None).
           eloss : 1d array, energy loss. If None, determined by evals (default: None).
             phi : float, azimuthal angle in unit of degree (default: 0).
            thin : 1d array or float, incident angle w.r.t. sample surface in unit of degree (default: 15).
             tth : float, scattering angle in unit of degree (default: 150).
    scatter_axis : 3x3 array, The local axis defining the scattering geometry. The scattering plane is defined in the local xz-plane.
                   It will be set to an identity matrix if None (default: None).
             pol : list of 2-element turple, x-ray polarization, (incident, emission), w.r.t the scattering plane.
              ei : 3-element list/array, incident polarization in sample coordinates.
                   If given, it will overwrite thin, tth, pol etc. (default: None).
              ef : 3-element list/array, emission polarization in sample coordinates.
                   If given, it will overwrite thin, tth, pol etc. (default: None).
               T : float, temperature in unit of Kelvin (default: 10).
             tol : float, tolerance of bose factor to include the initial states (default: 1e-3).
             ngs : int, number of initial states. If None it will be determined based on tol (default: None).

    Returns
    -------
    if ei/ef not given, rixs[pol,omega,thin,eloss] : 4d array.
    if ei/ef given, rixs[omega,eloss] : 2d array.

    Example
    -------
    >>> # rixs with sigma pol.
    >>> rixs = calc_RIXS_Q0(evals_i, evals_n, T_abs,
                            Gam_c=0.6, sigma=0.1, fraction=0.5,
                            omega=np.linspace(-2.9, 8.9, 100), eloss=np.linspace(-0.5, 10., 1000),
                            phi=0., thin=15., tth=150., pol=[(np.pi/2., 0), (np.pi/2., np.pi/2.)])
    >>> # rixs with pol. changed.
    >>> rixs = calc_RIXS_Q0(evals_i, evals_n, T_abs,
                            Gam_c=0.6, sigma=0.1, fraction=0.5,
                            omega=np.linspace(-2.9, 8.9, 100), eloss=np.linspace(-0.5, 10., 1000),
                            ei=[1,0,0], ef=[0,1,0])
    """
    # Determine the ground state occupation
    GS_energy = np.min(evals_i) ## ground state energy
    beta = 1.0 / 8.6173303E-5 / T
    if ngs is None:
        choose = np.exp(-beta * (evals_i - GS_energy)) >= tol
        gs = np.arange(len(evals_i))[choose]
    else:
        gs = np.arange(ngs)
    prob = np.exp(-beta * (evals_i[gs] - GS_energy)) / np.sum(np.exp(-beta * (evals_i[gs] - GS_energy)))

    # Determine omega
    if omega is None:
        omega_min = np.min(evals_n) - GS_energy - 10.
        omega_max = np.max(evals_n) - GS_energy + 10.
        omega_step = Gam_c / 10
        omega = np.arange(omega_min, omega_max, omega_step)

    # Determine eloss
    if eloss is None:
        eloss_min = -1.
        eloss_max = np.max(evals_i) - GS_energy + 1.
        eloss_step = sigma / 10
        eloss = np.arange(eloss_min, eloss_max, eloss_step)

    # Determine scattering axis
    if scatter_axis is None:
        scatter_axis = np.diag([1.]*3)

    # Reformulate the params
    if isinstance(omega, list):
        omega = np.array(omega)
    elif not isinstance(omega, np.ndarray):
        omega = np.array([omega])
    if isinstance(thin, list):
        thin = np.array(thin)
    elif not isinstance(thin, np.ndarray):
        thin = np.array([thin])
    if not isinstance(pol, list):
        pol = [pol]
    sigma_g = sigma / np.sqrt(2.*np.log(2))

    #new addition 23.11.03
    sigmarel_g = sigmarel / np.sqrt(2.*np.log(2))

    T_emi = np.stack([np.conj(np.transpose(Tabs_tmp)) for Tabs_tmp in T_abs])

    # Calculate RIXS
    if ei is None or ef is None: ## ei/ef not given
        ## Initialize
        rixs = np.zeros((len(pol), len(omega), len(thin), len(eloss)), dtype=float)
        for nomega, om in enumerate(omega):
            F_fi = scattering_mat(evals_i, evals_n, T_abs[:, :, gs], T_emi, om, Gam_c)
            for npol, (alpha, beta) in enumerate(pol):
                for nth, th in enumerate(thin):
                    ei, ef = dipole_polvec_rixs(th/180.*np.pi, (-th+tth)/180.*np.pi, phi/180.*np.pi, alpha, beta)
                    ei = np.dot(scatter_axis, ei)
                    ef = np.dot(scatter_axis, ef)
                    F_mag = np.zeros((len(evals_i), len(gs)), dtype=complex)
                    for m in range(3):
                        for n in range(3):
                            F_mag[:, :] += ef[m] * F_fi[m, n] * ei[n]
                    for nngs, gs_id in enumerate(gs):
                        for n in range(len(evals_i)):
                            Gaussian = (1 - fraction) / sigma_g / np.sqrt(2. * np.pi) * np.exp(-(eloss - (evals_i[n] - evals_i[gs_id]))**2 / 2. / sigma_g**2)
                            Lorentzian = fraction / np.pi * sigma / ((eloss - (evals_i[n] - evals_i[gs_id]))**2 + sigma**2)
                            rixs[npol,nomega,nth,:] += prob[nngs] * np.abs(F_mag[n, gs_id])**2 * (Gaussian + Lorentzian)
    else: ## ei/ef given
        ei = np.array(ei) / np.linalg.norm(ei)
        ef = np.array(ef) / np.linalg.norm(ef)
        ## Initialize
        rixs = np.zeros((len(omega), len(eloss)), dtype=float)
        for nomega, om in enumerate(omega):
            F_fi = scattering_mat(evals_i, evals_n, T_abs[:, :, gs], T_emi, om, Gam_c)
            F_mag = np.zeros((len(evals_i), len(gs)), dtype=complex)
            for m in range(3):
                for n in range(3):
                    F_mag[:, :] += ef[m] * F_fi[m, n] * ei[n]
            for nngs, gs_id in enumerate(gs):
                for n in range(len(evals_i)):
                    #new addition 23.11.03:
                    Gaussian = ((1 - fraction) / (sigma_g+sigmarel_g*(evals_i[n] - evals_i[gs_id])) / np.sqrt(2. * np.pi) *
                    np.exp(-(eloss - (evals_i[n] - evals_i[gs_id]))**2 / 2. / (sigma_g+sigmarel_g*(evals_i[n] - evals_i[gs_id]))**2))
                    Lorentzian = (fraction / np.pi * (sigma+sigmarel*(evals_i[n] - evals_i[gs_id])) / ((eloss - (evals_i[n] - evals_i[gs_id]))**2 +
                                                                                                      (sigma+sigmarel*(evals_i[n] - evals_i[gs_id]))**2))

                    #Gaussian = (1 - fraction) / sigma_g / np.sqrt(2. * np.pi) * np.exp(-(eloss - (evals_i[n] - evals_i[gs_id]))**2 / 2. / sigma_g**2)
                    #Lorentzian = fraction / np.pi * sigma / ((eloss - (evals_i[n] - evals_i[gs_id]))**2 + sigma**2)
                    rixs[nomega, :] += prob[nngs] * np.abs(F_mag[n, nngs])**2 * (Gaussian + Lorentzian)

    return rixs

class RIXS_runner_NIPS3:
    def __init__(self, view, eloss_in, omres_in):
        self.view = view
        self.eloss = eloss_in
        self.omres = omres_in

    def __call__(self, params_in,ed_only=False):
        """ Calculate RIXS spectrum in single atom approximation for NiPS3 """
        F0_d = edrixs.get_F0('d', params_in['F2_dd'], params_in['F4_dd'])
        F0_dp = edrixs.get_F0('dp',params_in['G1_dp'], params_in['G3_dp'])

        params = dict(tenDq=params_in['tenDq'],
              soc_v_i=params_in['soc_v_i'], soc_v_n=params_in['soc_v_n'], soc_c=params_in['soc_c'],
              F0_dd=F0_d, F2_dd=params_in['F2_dd'], F4_dd=params_in['F4_dd'],
              F0_dp=F0_dp, F2_dp=params_in['F2_dp'], G1_dp=params_in['G1_dp'], G3_dp=params_in['G3_dp'])
        evals_i, evecs_i, evals_n, evecs_n, T_abs = perform_ED_1site(**params)

        if ed_only == True:
            #return  evals_i, evals_n, T_abs, evecs_i, evecs_n
            sptab2,sptab4,sptabtr=saveops(('d','p'),8)
            v_soc=(params['soc_v_i'],params['soc_v_n'])
            c_soc=params['soc_c']
            v_cfmat = edrixs.cf_cubic_d(1.0*params['tenDq'])

            slater_i = [F0_d, params['F2_dd'], params['F4_dd']]   # Fk for d
            slater_n = [
                F0_d, params['F2_dd'], params['F4_dd'],   # Fk for d
                F0_dp, params['F2_dp'],        # Fk for dp
                params['G1_dp'], params['G3_dp'],        # Gk for dp
                0.0, 0.0           # Fk for p
            ]
            slater = [slater_i, slater_n]
            with contextlib.redirect_stdout(None):
                out = ed_1v1c_py_full_double(('d','p'), sptab2,sptab4,sptabtr,shell_level=(0, 0), v_soc=v_soc,
                                c_soc=c_soc, v_noccu=8, slater=slater,tenDq=params['tenDq'])
            return out

        RIXS_params = dict(omega=np.linspace(848.5-857.15+params_in['xoffset'], 857.6-857.15+params_in['xoffset'],self.omres), Gam_c=params_in['Gam_c'],
                   eloss=self.eloss, sigma=params_in['sigma'], fraction=0.)

        ei0, ef10, ef20 = get_pol('LH', tth=150., thin=23., phi=90.)
        eip, ef1p, ef2p = get_pol('LH', tth=150., thin=23., phi=90.+120.) # twin
        eim, ef1m, ef2m = get_pol('LH', tth=150., thin=23., phi=90.-120.) # twin

        params2=dict(evals_i=evals_i, evals_n=evals_n, T_abs=T_abs,T=40);
        paramlist=[dict(**params2,**RIXS_params,ei=ei0,ef=ef10),dict(**params2,**RIXS_params,ei=ei0,ef=ef20),
               dict(**params2,**RIXS_params,ei=eip,ef=ef1p),dict(**params2,**RIXS_params,ei=eip,ef=ef2p),
               dict(**params2,**RIXS_params,ei=eim,ef=ef1m),dict(**params2,**RIXS_params,ei=eim,ef=ef2m)];

        r1 = self.view.map(lambda x: calc_RIXS_Q0(**x),paramlist)
        rixs_lh=np.tensordot(np.ones(6),np.array(r1.get()),axes=([0],[0]))

        return  rixs_lh.T/3