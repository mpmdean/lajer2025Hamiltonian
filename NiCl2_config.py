import edrixs_python as edrixs
import sys
sys.modules['edrixs'] = edrixs
import numpy as np

from RIXS_utils import getParams, RIXS_runner

from solvers_fast import *
from solvers_fast_double import *

from optEDRIXS import target_L1sum
orbit='3d'
edge='L3'

# occupation number on d shell in the initial state
n_occu=8

# reference spectrum
rixs_ref=np.loadtxt('NiCl2.numpy');

output_dir ='OptData'

# descriptions to be saved in log file
descr='NiCl2 experiment, atomic, d'+str(n_occu)+', '+edge+' edge, L1 divergence measure, sum normalized'
method="BayesianOptimization with out of the box hyperparameters"
tfunc="distance_neg"


temperature = 40  # in K

geometry = {
    # beam angles
    'thin' : 10*np.pi/180,
    'thout' : (140)*np.pi/180,
    'phi' : 0.,

    #polarization
    'pol_type_rixs' : [('linear', np.pi/2, 'linear', 0), ('linear', np.pi/2, 'linear', np.pi/2)]
}

resolution = {
    # incident frequency (relative) bounds and resolution
    'ombounds' : [0,3.11],
    'omref'    : 32,

    # energy loss bounds and resolution
    'eloss' : np.linspace(0.5,3.5,311)
}

data_str='NiCl2_L1sum_Final'
data_opt='_subseq_opt2'
#num_runs_global=2
#num_runs_local=3
#num_iters=10

num_runs_global=10
num_runs_local=6
num_iters=20

init_points = 10
#init_seed = 0

# get ion parameters from database
tenDq=1.8

off, v_soc, c_soc, slater, v_cfmat, gamma_c = getParams('Ni',orbit,n_occu,edge,tenDq)
info = edrixs.utils.get_atom_data('Ni', orbit,n_occu,edge)
edge_ene = info['edge_ene'][0]

gamma_f=0.066
gamma_c = gamma_c * 1.5

sptab2,sptab4,sptabtr=saveops(('d','p'),n_occu)

true_values=dict(tenDq = tenDq, soc_v_i = v_soc[0], soc_v_n = v_soc[1],
                 soc_c = c_soc, F0_dd = slater[1][0], F2_dd = slater[1][1],
                 F4_dd = slater[1][2], F0_dp = slater[1][3],
                 F2_dp = slater[1][4], G1_dp = slater[1][5],
                 G3_dp = slater[1][6], Gam_c = gamma_c, sigma = gamma_f, xoffset = 0)

# parameter bounds
pbounds = {'tenDq': (0.5,2.5),
           'F2_dd': (true_values['F2_dd']*0.1, true_values['F2_dd']*1.2),
           'F4_dd': (true_values['F4_dd']*0.1, true_values['F4_dd']*1.2),
           'F2_dp': (true_values['F2_dp']*0.1,true_values['F2_dp']*1.2),
           'G1_dp': (true_values['G1_dp']*0.1,true_values['G1_dp']*1.2),
           'G3_dp': (true_values['G3_dp']*0.1,true_values['G3_dp']*1.2),
           'xoffset':(-12,12)}

fixed_params = {
       'soc_v_i' : true_values['soc_v_i'],
       'soc_v_n' : true_values['soc_v_n'],
       'soc_c' : true_values['soc_c'],
       'Gam_c' : true_values['Gam_c'],
       'sigma' : true_values['sigma']
}

rixs_funV =  RIXS_runner(ed_1v1c_py_full_double, rixs_1v1c_py_double, n_occu, edge_ene, temperature, geometry, resolution, sptab2, sptab4, sptabtr)

def fun(tenDq, F2_dd, F4_dd, F2_dp, G1_dp, G3_dp, xoffset):
    params = {**locals(), **fixed_params}
    return target_L1sum(params,rixs_funV,rixs_ref)

# will be set by running script
rand_seed = np.zeros(num_runs_local, dtype=np.int32)

# parameters for optimization
threshold_factor = 1.2
max_eval_greedy = 3600

greedy_bounds = {
        'F2_dd' : [1.2234, 14.6808],
        'F2_dp' : [0, 7.41216],
        'F4_dd' : [0.7598, 9.1176],
        'G1_dp' : [0.0, 6.9444],
        'G3_dp' : [0.0, 3.15936],
        'tenDq' : [0.5, 2.5],
        'xoffset' : [-12, 12],
        'soc_v_i' : [0, 0.2],
        'soc_v_n' : [0, 0.2],
        'soc_c' : [6, 15]
        }

greedy_filters = [
#    { "lhs": "F2_dd",  "op": ">=", "rhs": "F4_dd"  },
    { "lhs": "tenDq",  "op": "<",  "rhs": 2.5      }
]

fixed_params_greedy = {
#       'soc_c' : 11.48471778,
       'Gam_c' : true_values['Gam_c'],
       'sigma' : true_values['sigma']
}
def funGreedy(F2_dd, F2_dp, F4_dd, G1_dp, G3_dp, tenDq, xoffset, soc_v_i,soc_v_n,soc_c):
#def funGreedy(F2_dd, F2_dp, F4_dd, G1_dp, G3_dp, tenDq, xoffset, soc_v_i,soc_v_n):

    params = {**locals(), **fixed_params_greedy}
    return -target_L1sum(params,rixs_funV,rixs_ref)

record=dict(name = data_str, optname=data_opt, num_runs_local = num_runs_local, num_runs_global = num_runs_global, init_points = init_points,
            init_seed = init_seed, run_ind = 0, num_iters = num_iters,
            true_values = true_values, pbounds = pbounds, method = method,
            tfunc = tfunc, descr = descr, output_dir = output_dir, threshold_factor=threshold_factor,
            max_eval_greedy=max_eval_greedy, greedy_bounds = greedy_bounds, greedy_filters = greedy_filters,
            rand_seed = rand_seed.tolist())
