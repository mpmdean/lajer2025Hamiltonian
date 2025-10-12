import edrixs_python as edrixs
import sys
sys.modules['edrixs'] = edrixs
import numpy as np

#fasten up computations with some parallelization
import ipyparallel as ipp

#these dependencies are from the NiPS3 project
from utils_from_NiPS import get_pol, perform_ED_1site, calc_RIXS_Q0
from utils_from_NiPS import RIXS_runner_NIPS3
from RIXS_utils import getParams
from optEDRIXS import run_bayesian_optimization, target_L1sum, target_All

rc = ipp.Cluster(n=6).start_and_connect_sync()
rc.ids
view=rc[:];
with view.sync_imports():
    import edrixs_python as edrixs
    import numpy as np
    from utils_from_NiPS import  get_pol, perform_ED_1site, calc_RIXS_Q0

orbit = '3d'
edge = 'L3'

# occupation number on d shell in the initial state
n_occu = 8

rixs_ref = np.loadtxt('NiPS3v2.numpy');

eloss = np.linspace(0.5, 2.0,151)
ombounds = [0.,12.0]
omres = 40

output_dir ='OptData'

# descriptions to be saved in log file
descr = ('NiPS3 experiment, atomic, d'+str(n_occu)+', '+edge+
       ' edge, L1 divergence measure, sum normalized')
method = "BayesianOptimization with out of the box hyperparameters"
tfunc = "distance_neg"

data_str = 'NiPS3_L1sum_Final'
data_opt='_subseq_opt2'

num_runs_global=10
num_runs_local=6
num_iters=1010

init_points = 10
init_seed = 0

#placeholder; not used
tenDq = 0.0

# get ion parameters from database
off, v_soc, c_soc, slater, v_cfmat, gamma_c = getParams('Ni',orbit,n_occu,
                                                        edge,tenDq)

# overwrite gamma_c with hand-picked value
gamma_c = 0.75
gamma_f = 0.03

true_values=dict(tenDq = tenDq, soc_v_i = v_soc[0], soc_v_n = v_soc[1],
                 soc_c = c_soc, F0_dd = slater[1][0], F2_dd = slater[1][1],
                 F4_dd = slater[1][2], F0_dp = slater[1][3],
                 F2_dp = slater[1][4], G1_dp = slater[1][5],
                 G3_dp = slater[1][6], Gam_c = gamma_c, sigma = gamma_f)

# parameter bounds
pbounds = {'tenDq': (0.5,4),
           'F2_dd': (true_values['F2_dd']*0.1, true_values['F2_dd']*1.2),
           'F4_dd': (true_values['F4_dd']*0.1, true_values['F4_dd']*1.2),
           'F2_dp': (true_values['F2_dp']*0.1, true_values['F2_dp']*1.2),
           'G1_dp': (true_values['G1_dp']*0.1, true_values['G1_dp']*1.2),
           'G3_dp': (true_values['G3_dp']*0.1, true_values['G3_dp']*1.2),
           'xoffset':(-12,12)}

rand_seed=np.zeros(num_runs_local,dtype=np.int32)

rixs_funV =  RIXS_runner_NIPS3(view,eloss,omres)

fixed_params = {
       'soc_v_i' : true_values['soc_v_i'],
       'soc_v_n' : true_values['soc_v_n'],
       'soc_c' : true_values['soc_c'],
       'Gam_c' : true_values['Gam_c'],
       'sigma' : true_values['sigma']
}

def fun(tenDq, F2_dd, F4_dd, F2_dp, G1_dp, G3_dp, xoffset):
    params = {**locals(), **fixed_params}
    return target_L1sum(params,rixs_funV,rixs_ref)

# parameters for optimization
threshold_factor = 1.15
max_eval_greedy = 3600

greedy_bounds = {
        'F2_dd' : [0.0, None],
        'F2_dp' : [0.0, None],
        'F4_dd' : [0.0, None],
        'G1_dp' : [0.0, None],
        'G3_dp' : [0.0, None],
        'tenDq' : [0.5, None],
        'xoffset' : [-12., None],
        'soc_v_i' : [0., None],
        'soc_v_n' : [0., None],
        'soc_c' : [0., None],
        'Gam_c' : [0.,None],
        }

greedy_filters = [
    { "lhs": "F2_dd",  "op": ">=", "rhs": "F4_dd"  },
    { "lhs": "tenDq",  "op": "<",  "rhs": 2.5      }
]

fixed_params_greedy = {
#       'soc_c' : 11.48471778,

       'sigma' : true_values['sigma']
}

def funGreedy(F2_dd, F2_dp, F4_dd, G1_dp, G3_dp, tenDq, xoffset, soc_v_i,soc_v_n,soc_c, Gam_c):
    params = {**locals(), **fixed_params_greedy}
    return -target_L1sum(params,rixs_funV,rixs_ref)

def funAll(F2_dd, F2_dp, F4_dd, G1_dp, G3_dp, tenDq, xoffset, soc_v_i,soc_v_n,soc_c, Gam_c):
    params = {**locals(), **fixed_params_greedy}
    return -target_All(params,rixs_funV,rixs_ref, eloss[1]-eloss[0],(ombounds[1]-ombounds[0])/(omres-1))


record=dict(name = data_str, optname=data_opt,  num_runs_local=num_runs_local, num_runs_global = num_runs_global,
            init_points = init_points,
            init_seed = 0, run_ind = 0, num_iters = num_iters,
            true_values = true_values, pbounds = pbounds, method = method,
            tfunc = tfunc, descr = descr, output_dir = output_dir, threshold_factor=threshold_factor,
            max_eval_greedy=max_eval_greedy, greedy_bounds = greedy_bounds, greedy_filters = greedy_filters,
            rand_seed = rand_seed.tolist())

#run_bayesian_optimization(record, fun)
