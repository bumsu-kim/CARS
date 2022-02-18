from __future__ import print_function
import numpy as np
import pandas as pd
import copy
import pycutest
from matplotlib import pyplot as plt
#from benchmarkfunctions import SparseQuadratic, MaxK
#from oracle import Oracle, Oracle_pycutest
import optimizers

import argparse
from os.path import exists

# from Algorithms.stp_optimizer import STPOptimizer
# from Algorithms.gld_optimizer import GLDOptimizer
# from Algorithms.SignOPT2 import SignOPT
# from Algorithms.scobo_optimizer import SCOBOoptimizer
# from Algorithms.CMA_2 import CMA

'''
Read the parameters from a file
'''


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-p", "--params", type=str, default="params1",
                        help = "Parameters file name")
args = vars(parser.parse_args())
print(args)                        
paramsfile = args['params']
param = dict()
with open('csv/' + paramsfile, 'r') as f:
    for line in f:
        key, val = line.split()
        if is_number(val):
            val = float(val)
        param[key] = val
"""
BENCHMARKING.
"""

'''
Add this to ENVIRONMENT VARIABLES (edit configurations):
PYCUTEST_CACHE=.;CUTEST=/usr/local/opt/cutest/libexec;MYARCH=mac64.osx.gfo;SIFDECODE=/usr/local/opt/sifdecode/libexec;MASTSIF=/usr/local/opt/mastsif/share/mastsif;ARCHDEFS=/usr/local/opt/archdefs/libexec
'''

# Find unconstrained, variable-dimension problems.
probs = pycutest.find_problems(constraints='U', userN=True)
# print(sorted(probs)).
#print('number of problems: ', len(probs))
#print(sorted(probs))

# Properties of problem ROSENBR.
print('\n')

#for problem in probs:
#    print(problem + ': ' + str(pycutest.problem_properties(problem)))

# all have degree = 2.


# ----------------------------------------------------------------------------------------------------------------------
# here, I'll have all of the FUNCTIONS.
def run_CARS_pycutest(problem, param):
    # CARS.
    print('RUNNING ALGORITHM CARS....')
    p = problem
    # direction_vector_type = 0  # original.
    # direction_vector_type = '1  # gaussian.
    direction_vector_type = 'UNIF'  # uniform from sphere.
    # direction_vector_type = 3  # rademacher.
    a_k = 0.001  # step-size.
    x0 = copy.copy(param['x0'])
    n = len(x0)  # problem dimension.
    '''
    p.obj(x, Gradient=False) -> method which evaluates the function at x.
    '''
    #oracle_stp = Oracle_pycutest(p.obj)  # comparison oracle.
    
    # Instantiate the CARS optimizer object
    r0 = 0.1 # initial sampling radius
    param['dim'] = n
    param['target_fval'] = None
    cars_orig = optimizers.CARS(param, y0 = x0, f = p.obj)
    # step.
    termination = False
    prev_evals = 0
    while termination is False:
        solution, func_value, evals, grad_norm_seq, status, termination = cars_orig.step()
        grad_norm = grad_norm_seq[-1]
        #print('current value: ', func_value[-1])
    print('solution: ', solution if len(solution)<=5 else solution[:5])
    print('Status: ', status)
    print('function evaluation at solution: ', func_value[-1])
    print('Grad norm at solution: ', grad_norm)
    print('Number of function evaluations: ', evals)
    return func_value, grad_norm_seq, evals, status

'''
***************************
'''
# number of unconstrained PyCUTEST problems:  117
# problems:
# ['ARGLINA', 'ARGLINB', 'ARGLINC', 'ARGTRIGLS', 'ARWHEAD', 'BDQRTIC', 'BOX',
# 'BOXPOWER', 'BROWNAL', 'BROYDN3DLS', 'BROYDN7D', 'BROYDNBDLS', 'BRYBND', 'CHAINWOO', 'CHNROSNB', 'CHNRSNBM',
# 'COSINE', 'CRAGGLVY', 'CURLY10', 'CURLY20', 'CURLY30', 'DIXMAANA', 'DIXMAANB', 'DIXMAANC', 'DIXMAAND', 'DIXMAANE',
# 'DIXMAANF', 'DIXMAANG', 'DIXMAANH', 'DIXMAANI', 'DIXMAANJ', 'DIXMAANK', 'DIXMAANL', 'DIXMAANM', 'DIXMAANN',
# 'DIXMAANO', 'DIXMAANP', 'DIXON3DQ', 'DQDRTIC', 'DQRTIC', 'EDENSCH', 'EIGENALS', 'EIGENBLS', 'EIGENCLS', 'ENGVAL1',
# 'ERRINROS', 'ERRINRSM', 'EXTROSNB', 'FLETBV3M', 'FLETCBV2', 'FLETCBV3', 'FLETCHBV', 'FLETCHCR', 'FMINSRF2',
# 'FMINSURF', 'FREUROTH', 'GENHUMPS', 'GENROSE', 'HILBERTA', 'HILBERTB', 'INDEF', 'INDEFM', 'INTEQNELS', 'LIARWHD',
# 'LUKSAN11LS', 'LUKSAN12LS', 'LUKSAN13LS', 'LUKSAN14LS', 'LUKSAN15LS', 'LUKSAN16LS', 'LUKSAN17LS', 'LUKSAN21LS',
# 'LUKSAN22LS', 'MANCINO', 'MODBEALE', 'MOREBV', 'MSQRTALS', 'MSQRTBLS', 'NCB20', 'NCB20B', 'NONCVXU2', 'NONCVXUN',
# 'NONDIA', 'NONDQUAR', 'NONMSQRT', 'OSCIGRAD', 'OSCIPATH', 'PENALTY1', 'PENALTY2', 'PENALTY3', 'POWELLSG', 'POWER',
# 'QUARTC', 'SBRYBND', 'SCHMVETT', 'SCOSINE', 'SCURLY10', 'SCURLY20', 'SCURLY30', 'SENSORS', 'SINQUAD', 'SPARSINE',
# 'SPARSQUR', 'SPMSRTLS', 'SROSENBR', 'SSBRYBND', 'SSCOSINE', 'TESTQUAD', 'TOINTGSS', 'TQUARTIC', 'TRIDIA', 'VARDIM',
# 'VAREIGVL', 'WATSON', 'WOODS', 'YATP1LS', 'YATP2LS']
'''
I'm going to choose one at random and see how well our 4/5 algorithms optimize it.
'''
# for problem in sorted(probs):
#     print(problem + ': ' + str(pycutest.problem_properties(problem)))
#     '''
#     p = pycutest.import_problem(problem)
#     print('x0 dimension: ', len(p.x0))
#     '''

# input parameters.
sorted_problems = sorted(probs)




if  exists('list_of_probs_under_100'):
    with open('list_of_probs_under_100', 'r') as f:
        probs_under_100 = [line.rstrip() for line in f]
else:
    with open('list_of_probs_under_100','w') as f:
        probs_under_100 = []
        for p in sorted_problems:
            prob = pycutest.import_problem(p)
            x0 = prob.x0
            print('dimension of input vector of FUNCTION ' + str(p) + ': ' + str(len(x0)))
            # only want <= 100.
            if len(x0) <= 100:
                probs_under_100.append(p)
                f.write(p + '\n')
f.close()

print('\n')
print('number of problems with dimension = 100 or less: ', len(probs_under_100))
# should be 21.
# now, I want to iterate through PROBS_UNDER_100 list to create the graph.



num_experiments = 3
CARS_err_list = [[] for _ in range(num_experiments)]
CARS_gnorm_list = [[] for _ in range(num_experiments)]
CARS_evals_list = [[] for _ in range(num_experiments)]
CARS_status_list = [[] for _ in range(num_experiments)]




# for problem in list_of_problems_testing:
for problem in probs_under_100:
    for i in range(num_experiments):
        p_invoke_ = pycutest.import_problem(problem)
        '''
        x0_p_ = p_invoke_.x0
        dim_x0_ = len(x0_p_)
        print('dimension of problem: ', dim_x0_)
        x0_invoke_ = np.random.randn(dim_x0_)
        '''
        x0_invoke_ = p_invoke_.x0
        print('problem name: ', p_invoke_.name)
        print('dimension of problem: ', len(x0_invoke_))
        # STP.
        print('invoking CARS in a loop....')
        param['x0'] =  copy.copy(x0_invoke_)
        param['budget'] = param['budget_param']*len(x0_invoke_)
        fvals, gnorms, evals, status = run_CARS_pycutest(p_invoke_, param)
        CARS_err_list[i].append(fvals)
        CARS_gnorm_list[i].append(gnorms)
        CARS_evals_list[i].append(evals)
        CARS_status_list[i].append(status)
        print('\n')
        # # GLD.
        # print('invoking GLD in a loop....')
        # min2 = run_GLD_pycutest(p_invoke_, copy.copy(x0_invoke_), function_budget_)
        # GLD_err_list[i].append(min2)
        # print('\n')
        # # SignOPT.
        # print('invoking SignOPT in a loop....')
        # min3 = run_signOPT_pycutest(p_invoke_, copy.copy(x0_invoke_), function_budget_)
        # SignOPT_err_list[i].append(min3)
        # print('\n')
        # # SCOBO.
        # print('invoking SCOBO in a loop....')
        # min4 = run_SCOBO_pycutest(p_invoke_, copy.copy(x0_invoke_), function_budget_)
        # SCOBO_err_list[i].append(min4)
        # print('\n')
        # # CMA.
        # print('invoking CMA in a loop....')
        # min5 = run_CMA_pycutest(p_invoke_, copy.copy(x0_invoke_), function_budget_)
        # CMA_err_list[i].append(min5)
        # print('\n')

# averaging reference....
"""
multiple_lists = [[2,5,1,9], [4,9,5,10]]
arrays = [np.array(x) for x in multiple_lists]
[np.mean(k) for k in zip(*arrays)]
"""
# average CARS.
arrays_CARS = [np.array(x[-1]) for x in CARS_err_list]
CARS_average_error = [np.mean(k) for k in zip(*arrays_CARS)]
CARS_average_gnorm = [np.mean(k) for k in zip(*[np.array(x[-1]) for x in CARS_gnorm_list])]
CARS_average_evals = [np.mean(k) for k in zip(*[np.array(x) for x in CARS_evals_list])]
# # average GLD.
# arrays_gld = [np.array(x) for x in GLD_err_list]
# GLD_average_error = [np.mean(k) for k in zip(*arrays_gld)]
# # average SignOPT.
# arrays_signopt = [np.array(x) for x in SignOPT_err_list]
# SignOPT_average_error = [np.mean(k) for k in zip(*arrays_signopt)]
# # average SCOBO.
# arrays_scobo = [np.array(x) for x in SCOBO_err_list]
# SCOBO_average_error = [np.mean(k) for k in zip(*arrays_scobo)]
# # average CMA.
# arrays_cma = [np.array(x) for x in CMA_err_list]
# CMA_average_error = [np.mean(k) for k in zip(*arrays_cma)]

'''
print('STP: ', STP_err_list)
print('GLD: ', GLD_err_list)
print('SignOPT: ', SignOPT_err_list)
print('SCOBO: ', SCOBO_err_list)
'''
print('CARS: ', CARS_average_error)
# print('GLD: ', GLD_average_error)
# print('SignOPT: ', SignOPT_average_error)
# print('SCOBO: ', SCOBO_average_error)
# print('CMA: ', CMA_average_error)

# list_of_errors = [STP_err_list, GLD_err_list, SignOPT_err_list, SCOBO_err_list]
list_of_errors = [CARS_average_error,]# GLD_average_error, SignOPT_average_error, SCOBO_average_error, CMA_average_error]
list_of_gnorm = [CARS_average_gnorm,]# GLD_average_error, SignOPT_average_error, SCOBO_average_error, CMA_average_error]
list_of_evals = [CARS_average_evals,]# GLD_average_error, SignOPT_average_error, SCOBO_average_error, CMA_average_error]
list_of_algorithms = ['CARS',] # 'GLD', 'SignOPT', 'SCOBO', 'CMA']

# I need to make a dataframe with rows = Algorithms and columns = problems.
# columns = [element for element in list_of_problems_testing]
columns = [element for element in probs_under_100]
df_err = pd.DataFrame(columns=columns)
df_gnorm = pd.DataFrame(columns=columns)
df_evals = pd.DataFrame(columns=columns)
df_length = len(df_err)
for i in range(len(list_of_errors)):
    df_err.loc[list_of_algorithms[i]] = list_of_errors[i]
    df_gnorm.loc[list_of_algorithms[i]] = list_of_gnorm[i]
    df_evals.loc[list_of_algorithms[i]] = list_of_evals[i]

print("Errors:")
print(df_err)
print(type(df_err))
print("Evals:")
print(df_evals)


path_name_err = "csv/CARS_DF_err_" + paramsfile + ".csv"
path_name_gnorm = "csv/CARS_DF_gnorm_" + paramsfile + ".csv"
path_name_evals = "csv/CARS_DF_evals_" + paramsfile + ".csv"
df_err.to_csv(path_name_err)
df_evals.to_csv(path_name_evals)
df_gnorm.to_csv(path_name_gnorm)

for i in range(num_experiments):
    j = 0
    for p in probs_under_100:
        with open('npy/CARS_err_gnorm_'+paramsfile+f'_exp_{i}_{p}.npy', 'w') as f:
            np.save(f, np.vstack((CARS_err_list[i][j], CARS_gnorm_list[i][j])))
        j = j+1

# trying something interesting....

# x0_p = p_invoke.x0
# dim_x0 = len(x0_p)
# x0_invoke = np.random.randn(dim_x0)

# #x0_invoke = p_invoke.x0  # initial value.
# print('dimension of x0: ', len(x0_invoke))
# # x0_invoke = np.random.randn(2)
# function_budget = 10  # max number of iterations.
# # STP invocation.
# print('\n')
# print('invoke STP as a function.')
# #run_STP_pycutest(p_invoke, copy.copy(x0_invoke), function_budget)
# # GLD invocation.
# print('\n')
# print('invoke GLD as a function.')
# #run_GLD_pycutest(p_invoke, copy.copy(x0_invoke), function_budget)
# # SignOPT invocation.
# print('\n')
# print('invoke SignOPT as a function.')
# #run_signOPT_pycutest(p_invoke, copy.copy(x0_invoke), function_budget)
# # SCOBO invocation.
# print('\n')
# print('invoke SCOBO as a function.')
# #run_SCOBO_pycutest(p_invoke, copy.copy(x0_invoke), function_budget)
'''
***************************
'''