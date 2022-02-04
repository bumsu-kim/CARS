from __future__ import print_function
import numpy as np
import pandas as pd
import copy
import pycutest
from matplotlib import pyplot as plt
#from benchmarkfunctions import SparseQuadratic, MaxK
#from oracle import Oracle, Oracle_pycutest
import optimizers

# from Algorithms.stp_optimizer import STPOptimizer
# from Algorithms.gld_optimizer import GLDOptimizer
# from Algorithms.SignOPT2 import SignOPT
# from Algorithms.scobo_optimizer import SCOBOoptimizer
# from Algorithms.CMA_2 import CMA


"""
BENCHMARKING.
"""

# email this to Daniel:
'''
Add this to ENVIRONMENT VARIABLES (edit configurations):
PYCUTEST_CACHE=.;CUTEST=/usr/local/opt/cutest/libexec;MYARCH=mac64.osx.gfo;SIFDECODE=/usr/local/opt/sifdecode/libexec;MASTSIF=/usr/local/opt/mastsif/share/mastsif;ARCHDEFS=/usr/local/opt/archdefs/libexec
'''

# Find unconstrained, variable-dimension problems.
probs = pycutest.find_problems(constraints='U', userN=True)
# print(sorted(probs)).
print('number of problems: ', len(probs))
print(sorted(probs))

# Properties of problem ROSENBR.
print('\n')

for problem in probs:
    print(problem + ': ' + str(pycutest.problem_properties(problem)))

# all seem to have degree = 2.

five_probs = sorted(probs[:5])
print(five_probs)

# IDEA:
'''
1. run ROSENBR for each of the 5 algorithms (STP, GLD, SignOPT, SCOBO, CMA).
for each of the five problems in five_probs, generate a plot. 
each plot will have 5 lines (one for each algorithm we are benchmarking against). 
start with the same x0 for each.
make sure you can run each of these algorithms against a pycutest problem.
then, at first, automatically save these plots.
then, using your experience from McGraw Hill (go through old internship code), figure out how to automatically 
save the plots.
have your code generate a folder automatically (made different by appending the current DATE to the name).
put each of the 117 plots into a folder; send to Daniel.
by the end of TODAY: have the code running for 5 algorithms plotting....
'''
#
# # testing with ROSENBR.
p = pycutest.import_problem('ROSENBR')  # objective function.
x0 = p.x0  # initial value.
n = len(x0)  # problem dimension.
function_budget = 1000  # max number of iterations.
#
# # ---------------------------
# # STP.
# print('RUNNING ALGORITHM STP....')
# # direction_vector_type = 0  # original.
# # direction_vector_type = 1  # gaussian.
# direction_vector_type = 2  # uniform from sphere.
# # direction_vector_type = 3  # rademacher.
# a_k = 0.001  # step-size.
# x0_stp = copy.copy(x0)
# '''
# p.obj(x, Gradient=False) -> method which evaluates the function at x.
# '''
# oracle_stp = Oracle_pycutest(p.obj)  # comparison oracle.
# # STP instance.
# stp1 = STPOptimizer(oracle_stp, direction_vector_type, x0_stp, n, a_k, p.obj, 2 * function_budget)
# # step.
# termination = False
# prev_evals = 0
# while termination is False:
#     solution, func_value, termination = stp1.step()
#     print('current value: ', func_value[-1])
# print('solution: ', solution)
# # plot.
# plt.plot(func_value)
# plt.title('STP - linear.')
# plt.show()
# plt.close()
# plt.semilogy(func_value)
# plt.title('STP - log.')
# plt.show()
# plt.close()
#
# # ---------------------------
# # GLD.
# print('RUNNING ALGORITHM GLD....')
# R_ = 1e-1
# r_ = 1e-4
# x0_gld = copy.copy(x0)
# '''
# p.obj(x, Gradient=False) -> method which evaluates the function at x.
# '''
# oracle_gld = Oracle_pycutest(p.obj)  # comparison oracle.
# # GLD instance.
# gld1 = GLDOptimizer(oracle_gld, p.obj, x0_gld, R_, r_, 2 * function_budget)
# # step.
# termination = False
# prev_evals = 0
# while termination is False:
#     solution, func_value, termination = gld1.step()
#     print('current value: ', func_value[-1])
# print('solution: ', solution)
# # plot.
# plt.plot(func_value)
# plt.title('GLD - linear.')
# plt.show()
# plt.close()
# plt.semilogy(func_value)
# plt.title('GLD - log.')
# plt.show()
# plt.close()
#
# # ---------------------------
# # SignOPT.
# print('RUNNING ALGORITHM SIGNOPT....')
# m = 100
# x0_signopt = copy.copy(x0)
# step_size = 0.2
# r = 0.1
# '''
# p.obj(x, Gradient=False) -> method which evaluates the function at x.
# '''
# oracle_signopt = Oracle_pycutest(p.obj)  # comparison oracle.
# # signOPT instance.
# signopt1 = SignOPT(oracle_signopt, function_budget, x0_signopt, m, step_size, r, debug=False, function=p.obj)
# # step.
# for i in range(function_budget - 1):
#     print(i)
#     signopt1.step()
# # plot.
# plt.plot(signopt1.f_vals)
# plt.title('SignOPT - linear.')
# plt.show()
# plt.close()
# plt.semilogy(signopt1.f_vals)
# plt.title('SignOPT - log.')
# plt.show()
# plt.close()
#
# # ---------------------------
# # SCOBO.
# print('RUNNING ALGORITHM SCOBO....')
# m_scobo = 100  # should always be larger than s_exact.
# x0_scobo = copy.copy(x0)
# stepsize = 0.01
# s_exact = 20
# r = 0.1
# '''
# p.obj(x, Gradient=False) -> method which evaluates the function at x.
# '''
# oracle_scobo = Oracle_pycutest(p.obj)  # comparison oracle.
# # SCOBO instance.
# scobo1 = SCOBOoptimizer(oracle_scobo, stepsize, function_budget, x0_scobo, r, m_scobo, s_exact, objfunc=p.obj)
# # step.
# for i in range(function_budget):
#     print(i)
#     err = scobo1.step()
#     print(err)
# # plot.
# plt.plot(scobo1.function_vals)
# plt.title('SCOBO - linear.')
# plt.show()
# plt.close()
# plt.semilogy(scobo1.function_vals)
# plt.title('SCOBO - log.')
# plt.show()
# plt.close()

# ---------------------------

"""
# CMA.
m_cma = 100
x0_cma = copy.copy(x0)
step_size_cma = 0.2
r = 0.1
lam = 10
mu = 1
sigma = 0.5
'''
p.obj(x, Gradient=False) -> method which evaluates the function at x.
'''
oracle_cma = Oracle_pycutest(p.obj)  # comparison oracle.
# CMA instance.
all_func_vals = []
cma1 = CMA(oracle_cma, function_budget, x0_cma, lam, mu, sigma, function=p.obj)
# step.
for i in range(function_budget):
    val = cma1.step()
    print(str(i) + ': ' + str(val))
    # handling error of convergence.
    if i > 1:
        if np.abs(val - all_func_vals[-1]) < 1e-6:
            all_func_vals.append(val)
            break
    all_func_vals.append(val)
# plot.
plt.plot(all_func_vals)
plt.title('CMA - linear.')
plt.show()
plt.close()
plt.semilogy(all_func_vals)
plt.title('CMA - log.')
plt.show()
plt.close()
"""



# ----------------------------------------------------------------------------------------------------------------------
# here, I'll have all of the FUNCTIONS.
def run_CARS_pycutest(problem, x0, function_budget):
    # CARS.
    print('RUNNING ALGORITHM CARS....')
    p = problem
    # direction_vector_type = 0  # original.
    # direction_vector_type = '1  # gaussian.
    direction_vector_type = 'UNIF'  # uniform from sphere.
    # direction_vector_type = 3  # rademacher.
    a_k = 0.001  # step-size.
    x0_stp = copy.copy(x0)
    n = len(x0_stp)  # problem dimension.
    '''
    p.obj(x, Gradient=False) -> method which evaluates the function at x.
    '''
    #oracle_stp = Oracle_pycutest(p.obj)  # comparison oracle.
    
    # Instantiate the CARS optimizer object
    r0 = 0.001 # initial sampling radius
    param = {
        'dim': n, # problem dim
        'r': r0, # sampling radius
        'budget': function_budget,
        'nq': 0, # not CARS-NQ
        'dist_dir': 'UNIF',
        'target_fval': None, # no min vals known
        'threshold_first_order_opt': 1e-10 # use first order optimality condition instead
    }
    cars_orig = optimizers.CARS(param, y0 = x0, f = p.obj)
    # step.
    termination = False
    prev_evals = 0
    while termination is False:
        solution, func_value, termination = cars_orig.step()
        #print('current value: ', func_value[-1])
    print('solution: ', solution)
    # plot.
    plt.plot(func_value)
    plt.title('CARS - linear.')
    #plt.show()
    plt.close()
    plt.semilogy(func_value)
    plt.title('CARS - log.')
    #plt.show()
    plt.close()
    print('function evaluation at solution: ', func_value[-1])
    return func_value[-1]


# def run_GLD_pycutest(problem, x0, function_budget):
#     # GLD.
#     print('RUNNING ALGORITHM GLD....')
#     p = problem
#     R_ = 1e-1
#     r_ = 1e-4
#     x0_gld = copy.copy(x0)
#     n = len(x0_gld)  # problem dimension.
#     '''
#     p.obj(x, Gradient=False) -> method which evaluates the function at x.
#     '''
#     oracle_gld = Oracle_pycutest(p.obj)  # comparison oracle.
#     # GLD instance.
#     gld1 = GLDOptimizer(oracle_gld, p.obj, x0_gld, R_, r_, 2 * function_budget)
#     # step.
#     termination = False
#     prev_evals = 0
#     while termination is False:
#         solution, func_value, termination = gld1.step()
#         #print('current value: ', func_value[-1])
#     print('solution: ', solution)
#     # plot.
#     plt.plot(func_value)
#     plt.title('GLD - linear.')
#     #plt.show()
#     plt.close()
#     plt.semilogy(func_value)
#     plt.title('GLD - log.')
#     #plt.show()
#     plt.close()
#     print('function evaluation at solution: ', func_value[-1])
#     return func_value[-1]


# def run_signOPT_pycutest(problem, x0, function_budget):
#     # SignOPT.
#     print('RUNNING ALGORITHM SIGNOPT....')
#     p = problem
#     '''
#     m = 100
#     '''
#     # new value:
#     # tune hyperparameter.
#     m = 10
#     x0_signopt = copy.copy(x0)
#     n = len(x0_signopt)  # problem dimension.
#     step_size = 0.2
#     r = 0.1
#     '''
#     p.obj(x, Gradient=False) -> method which evaluates the function at x.
#     '''
#     oracle_signopt = Oracle_pycutest(p.obj)  # comparison oracle.
#     # signOPT instance.
#     signopt1 = SignOPT(oracle_signopt, function_budget, x0_signopt, m, step_size, r, debug=False, function=p.obj)
#     # step.
#     for i in range(function_budget - 1):
#         #print(i)
#         signopt1.step()
#     # plot.
#     plt.plot(signopt1.f_vals)
#     plt.title('SignOPT - linear.')
#     #plt.show()
#     plt.close()
#     plt.semilogy(signopt1.f_vals)
#     plt.title('SignOPT - log.')
#     #plt.show()
#     plt.close()
#     print('function evaluation at solution: ', signopt1.f_vals[-1])
#     return signopt1.f_vals[-1]

# def run_SCOBO_pycutest(problem, x0, function_budget):
#     # SCOBO.
#     print('RUNNING ALGORITHM SCOBO....')
#     p = problem
#     m_scobo = 100  # should always be larger than s_exact.
#     x0_scobo = copy.copy(x0)
#     n = len(x0_scobo)  # problem dimension.
#     stepsize = 0.01
#     # s_exact = 20
#     s_exact = 0.1*n
#     r = 0.1
#     '''
#     p.obj(x, Gradient=False) -> method which evaluates the function at x.
#     '''
#     oracle_scobo = Oracle_pycutest(p.obj)  # comparison oracle.
#     # SCOBO instance.
#     scobo1 = SCOBOoptimizer(oracle_scobo, stepsize, function_budget, x0_scobo, r, m_scobo, s_exact, objfunc=p.obj)
#     # step.
#     for i in range(function_budget):
#         #print(i)
#         err = scobo1.step()
#         #print(err)
#     # plot.
#     plt.plot(scobo1.function_vals)
#     plt.title('SCOBO - linear.')
#     #plt.show()
#     plt.close()
#     plt.semilogy(scobo1.function_vals)
#     plt.title('SCOBO - log.')
#     #plt.show()
#     plt.close()
#     print('function evaluation at solution: ', scobo1.function_vals[-1])
#     return scobo1.function_vals[-1]

# def run_CMA_pycutest(problem, x0, function_budget):
#     # CMA.
#     print('RUNNING ALGORITHM CMA....')
#     p = problem
#     m_cma = 100
#     x0_cma = copy.copy(x0)
#     step_size_cma = 0.2
#     r = 0.1
#     lam = 10
#     mu = 1
#     sigma = 0.5
#     '''
#     p.obj(x, Gradient=False) -> method which evaluates the function at x.
#     '''
#     oracle_cma = Oracle_pycutest(p.obj)  # comparison oracle.
#     # CMA instance.
#     all_func_vals = []
#     cma1 = CMA(oracle_cma, function_budget, x0_cma, lam, mu, sigma, function=p.obj)
#     # step.
#     for ij in range(function_budget):
#         val = cma1.step()
#         print(str(ij) + ': ' + str(val))
#         # handling error of convergence.
#         if ij > 1:
#             if np.abs(val - all_func_vals[-1]) < 1e-6:
#                 all_func_vals.append(val)
#                 break
#         all_func_vals.append(val)
#     # plot.
#     plt.plot(all_func_vals)
#     plt.title('CMA - linear.')
#     #plt.show()
#     plt.close()
#     plt.semilogy(all_func_vals)
#     plt.title('CMA - log.')
#     #plt.show()
#     plt.close()
#     print('function evaluation at solution: ', all_func_vals[-1])
#     return all_func_vals[-1]


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
for problem in sorted(probs):
    print(problem + ': ' + str(pycutest.problem_properties(problem)))
    '''
    p = pycutest.import_problem(problem)
    print('x0 dimension: ', len(p.x0))
    '''

# input parameters.
"""
invoke_problem = 'ROSENBR'
"""
invoke_problem = 'COSINE'
'PYCUTEST_CACHE=.;CUTEST=/usr/local/opt/cutest/libexec;MYARCH=mac64.osx.gfo;SIFDECODE=/usr/local/opt/sifdecode/libexec;MASTSIF=/usr/local/opt/mastsif/share/mastsif'
p_invoke = pycutest.import_problem(invoke_problem)  # objective function.

'''
list_of_problems_testing = ['ROSENBR', 'COSINE', 'FLETCBV3']
'''
# five_probs = sorted(probs[:5])
# print(five_probs)
sorted_problems = sorted(probs)
list_of_problems_testing = sorted_problems[:10]

probs_under_100 = []

#for p in sorted_problems:
for p in sorted_problems[:30]: 
    prob = pycutest.import_problem(p)
    x0 = prob.x0
    print('dimension of input vector of FUNCTION ' + str(p) + ': ' + str(len(x0)))
    # only want <= 100.
    if len(x0) <= 100:
        probs_under_100.append(p)


print('\n')
print('number of problems with dimension = 100 or less: ', len(probs_under_100))
# should be 21.
# now, I want to iterate through PROBS_UNDER_100 list to create the graph.

CARS_err_1 = []
CARS_err_2 = []
CARS_err_3 = []
CARS_err_4 = []
CARS_err_5 = []
CARS_err_list = [CARS_err_1, CARS_err_2, CARS_err_3, CARS_err_4, CARS_err_5]
# gld_err_1 = []
# gld_err_2 = []
# gld_err_3 = []
# gld_err_4 = []
# gld_err_5 = []
# GLD_err_list = [gld_err_1, gld_err_2, gld_err_3, gld_err_4, gld_err_5]
# signopt_err_1 = []
# signopt_err_2 = []
# signopt_err_3 = []
# signopt_err_4 = []
# signopt_err_5 = []
# SignOPT_err_list = [signopt_err_1, signopt_err_2, signopt_err_3, signopt_err_4, signopt_err_5]
# scobo_err_1 = []
# scobo_err_2 = []
# scobo_err_3 = []
# scobo_err_4 = []
# scobo_err_5 = []
# SCOBO_err_list = [scobo_err_1, scobo_err_2, scobo_err_3, scobo_err_4, scobo_err_5]
# cma_err_1 = []
# cma_err_2 = []
# cma_err_3 = []
# cma_err_4 = []
# cma_err_5 = []
# CMA_err_list = [cma_err_1, cma_err_2, cma_err_3, cma_err_4, cma_err_5]

# for problem in list_of_problems_testing:
for problem in probs_under_100:
    for i in range(5):
        p_invoke_ = pycutest.import_problem(problem)
        '''
        x0_p_ = p_invoke_.x0
        dim_x0_ = len(x0_p_)
        print('dimension of problem: ', dim_x0_)
        x0_invoke_ = np.random.randn(dim_x0_)
        '''
        x0_invoke_ = p_invoke_.x0
        print('dimension of problem: ', len(x0_invoke_))
        function_budget_ = 100
        # STP.
        print('invoking CARS in a loop....')
        min1 = run_CARS_pycutest(p_invoke_, copy.copy(x0_invoke_), function_budget_)
        CARS_err_list[i].append(min1)
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
arrays_CARS = [np.array(x) for x in CARS_err_list]
CARS_average_error = [np.mean(k) for k in zip(*arrays_CARS)]
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
list_of_algorithms = ['CARS',] # 'GLD', 'SignOPT', 'SCOBO', 'CMA']

# I need to make a dataframe with rows = Algorithms and columns = problems.
# columns = [element for element in list_of_problems_testing]
columns = [element for element in probs_under_100]
df = pd.DataFrame(columns=columns)
df_length = len(df)
for i in range(len(list_of_errors)):
    df.loc[list_of_algorithms[i]] = list_of_errors[i]

print(df)
print(type(df))

path_name = "csv/CARS_DF.csv"
df.to_csv(path_name)

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