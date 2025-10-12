import edrixs_python as edrixs
import sys
sys.modules['edrixs'] = edrixs

import numpy as np
from solvers_fast import *
from solvers_fast_double import *
# suppress stdout output
import contextlib

def numgrad(cmap,dEloss,dOm):
    dycmap=(cmap[1:]-cmap[:-1])/(dEloss)
    dxcmap=(cmap[:,1:]-cmap[:,:-1])/(dOm)
    return np.sqrt(dycmap[:,:-1]**2+dxcmap[:-1]**2)

def rmat():
    """
    define the tranformation matrix from
    """
    rmat = np.zeros((3,3), dtype=float)
    rmat[:,0] = [np.sqrt(1/6), -np.sqrt(1/2), np.sqrt(1/3)]
    rmat[:,1] = [np.sqrt(1/6), np.sqrt(1/2), np.sqrt(1/3)]
    rmat[:,2] = [-np.sqrt(2/3), 0., np.sqrt(1/3)]
    return rmat.T

def getParams(elem,orbit,n_occu,edge,tenDq):
    info = edrixs.utils.get_atom_data(elem, orbit,n_occu,edge='L3')
    infosl=dict(info['slater_n'])
    F0_d = edrixs.get_F0('d', infosl['F2_11'], infosl['F4_11'])
    F0_dp = edrixs.get_F0('dp',infosl['G1_12'], infosl['G3_12'])
    off = info['edge_ene'][0] - n_occu*F0_d+2/5*tenDq
    v_cfmat = edrixs.cf_cubic_d(tenDq)
    v_soc = (info['v_soc_i'][0],info['v_soc_n'][0])
    c_soc = info['c_soc']
    g_c = info['gamma_c'][0]
    slater_i = [F0_d, infosl['F2_11'], infosl['F4_11']]   # Fk for d
    slater_n = [
        F0_d, infosl['F2_11'], infosl['F4_11'],   # Fk for d
        F0_dp, infosl['F2_12'],        # Fk for dp
        infosl['G1_12'], infosl['G3_12'],        # Gk for dp
        0.0, 0.0           # Fk for p
    ]
    slater = [slater_i, slater_n]
    return off, v_soc, c_soc, slater, v_cfmat, g_c

def getParamsFromTV(trueValues,n_occu):
    tenDq = trueValues['tenDq']
    #F0_d = trueValues['F0_dd']
    #F0_dp = trueValues['F0_dp']
    F0_d = edrixs.get_F0('d', trueValues['F2_dd'], trueValues['F4_dd'])
    F0_dp = edrixs.get_F0('dp',trueValues['G1_dp'], trueValues['G3_dp'])
    off =  - n_occu*F0_d+2/5*tenDq
    v_cfmat = edrixs.cf_cubic_d(tenDq)
    v_soc = (trueValues['soc_v_i'],trueValues['soc_v_n'])
    c_soc = trueValues['soc_c']
    g_c = trueValues['Gam_c']
    slater_i = [F0_d, trueValues['F2_dd'], trueValues['F4_dd']]   # Fk for d
    slater_n = [
        F0_d, trueValues['F2_dd'], trueValues['F4_dd'],   # Fk for d
        F0_dp, trueValues['F2_dp'],        # Fk for dp
        trueValues['G1_dp'], trueValues['G3_dp'],        # Gk for dp
        0.0, 0.0           # Fk for p
    ]
    slater = [slater_i, slater_n]
    return off, v_soc, c_soc, slater, v_cfmat, g_c

class RIXS_runner:
    def __init__(self, ED_fun, RIXS_fun, n_occu, edge_ene, temperature, geometry, resolution,
                  sptab2 = None,sptab4 = None,sptabtr = None,
                 vshell='d', cshell='p', ext_B=(0,0,0), is_powder=False):
        self.ED_fun = ED_fun
        self.RIXS_fun = RIXS_fun
        self.temperature = temperature
        self.geometry = geometry
        self.resolution = resolution
        self.n_occu = n_occu
        self.edge_ene = edge_ene
        self.sptab2  = sptab2
        self.sptab4  = sptab4
        self.sptabtr = sptabtr
        self.vshell = vshell
        self.cshell = cshell
        self.ext_B = ext_B
        self.is_powder = is_powder

    def __call__(self, params_in,*,ed_only=False):
        F0_d = edrixs.get_F0('d', params_in['F2_dd'], params_in['F4_dd'])
        F0_dp = edrixs.get_F0('dp',params_in['G1_dp'], params_in['G3_dp'])
        off = self.edge_ene - self.n_occu*F0_d+2/5*params_in['tenDq']
        v_cfmat = edrixs.cf_cubic_d(params_in['tenDq'])
        if params_in['soc_v_n'] is not None:
            v_soc =(params_in['soc_v_i'],params_in['soc_v_n'])
        else:
            v_soc =(params_in['soc_v_i'],params_in['soc_v_i'])

        slater=[[F0_d, params_in['F2_dd'], params_in['F4_dd']],
             [F0_d, params_in['F2_dd'], params_in['F4_dd'],
              F0_dp, params_in['F2_dp'],        # Fk for dp
              params_in['G1_dp'], params_in['G3_dp'],        # Gk for dp
              0.0, 0.0]]

        with contextlib.redirect_stdout(None):
            if self.sptab2 is not None:
                out = self.ED_fun((self.vshell,self.cshell),self.sptab2,self.sptab4,self.sptabtr, shell_level=(0, -off), v_soc=v_soc,
                                c_soc=params_in['soc_c'], v_noccu=self.n_occu, slater=slater,tenDq=params_in['tenDq'],ext_B=self.ext_B)
                eval_i, eval_n, trans_op = out[:3]
            else:
                out = self.ED_fun((self.vshell,self.cshell), shell_level=(0, -off), v_soc=v_soc,
                                c_soc=params_in['soc_c'], v_noccu=self.n_occu, slater=slater,tenDq=params_in['tenDq'],ext_B=self.ext_B)
                eval_i, eval_n, trans_op = out[:3]

        if ed_only:
            return out

        ominc = np.linspace(self.resolution['ombounds'][0] + off, self.resolution['ombounds'][1] + off, self.resolution['omref'])+params_in['xoffset']

        prob = edrixs.boltz_dist(eval_i, self.temperature)
        gs_list = [n for n, prob in enumerate(prob) if prob > 1e-6]

        with contextlib.redirect_stdout(None):
            rixs = self.RIXS_fun(
                eval_i, eval_n, trans_op, ominc, self.resolution['eloss'],
                gamma_c=params_in['Gam_c'], gamma_f=params_in['sigma'],
                thin=self.geometry['thin'], thout=self.geometry['thout'], phi=self.geometry['phi'],
                pol_type=self.geometry['pol_type_rixs'], gs_list=gs_list,
                temperature=self.temperature,isPowder=self.is_powder
            )
        return rixs.sum(-1).T
"""




NiCl2:
rixs_funV =  RIXS_runner(ed_1v1c_py_full, rixs_1v1c_py, n_occu, edge_ene, temperature, geometry, resolution, sptab2, sptab4, sptabtr)

Fe2O3:
rixs_funV =  RIXS_runner(ed_1v1c_py_full, rixs_1v1c_py, n_occu, edge_ene, temperature, geometry, resolution, sptab2, sptab4, sptabtr,ext_B=(0.033,0.0,0))

Ca3LiOsO6:
rixs_funV =  RIXS_runner(ed_1v1c_py_full, rixs_1v1c_py, n_occu, edge_ene, temperature, geometry, resolution, sptab2, sptab4, sptabtr, cshell='p32',is_powder = True)
"""