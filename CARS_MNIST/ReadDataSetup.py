
import os
from os import listdir
from os.path import isfile, join, isdir

'''
Usage:

exp_name : the name of the experiment (name it as you want)
testname : Name of the Folder containing the results
            (if not modified, this is determined by --name_to_save option in MNIST_Attack.py.
            Folder name is Results_(name_to_save)_(starting_tid) (see MINST_Attack.py, "savedir" variable)
            e.g. if options are given by [ -name "CARS_Test" -t 10 100 ],
                testname = "Results_CARS_Test_10"
            This is designed for easier gathering of results when
                you are running the algorithm in several different machines.
            In this case, you will get those folders with different tid's.
            Say these are A_t1, A_t2, ...
            Then create a folder B, and put all the folders A_ti's in B.
                i.e. B/A_t1/, B/A_t2, ...
            In this case, testname = B.
onames   : Name of the algorithm you used in MNIST_Attack.py
            The result files should have [oname].csv

example file paths:
    ../CARS_MNIST/TEST1/CARS_TEST_0/CARS_TEST.csv
    ../CARS_MNIST/TEST1/CARS_TEST_350/CARS_TEST.csv
    ../CARS_MNIST/TEST1/CARS_TEST_700/CARS_TEST.csv

    Then use testname = 'TEST1', onames = ['CARS_TEST'].
'''
def Setup_Directories(exp_name, subdiropt = ""):
    
    '''
    The followings are examples of attacked data
    '''
    if exp_name == 'carsnq': # CARS-NQ with NQ=5
        testname = 'CARS_NQ'
        onames = ['CARS_nq_5'] 

    elif exp_name == 'cars_june15': # can add the date
        testname = 'CARS_June15'
        onames = ['CARS_June15']

    elif exp_name == 'carsbox0.3': # eps = 0.3
        testname = 'CARS_BOX_0.3'
        onames = ['CARS_BOX']

    elif exp_name == 'carsbox0.2': # eps = 0.2
        testname = 'CARS_BOX_0.2'
        onames = ['CARS_BOX_0.2_B1_p0.2'] # quite long oname used here

    elif exp_name == 'square0.2': # square attack, eps = 0.2
        testname = 'SQUARE_0.2'
        onames = ['square0.2_p0.8']

    elif exp_name == 'carsunif0.2': # cars with uniform dist. on sphere (instead of squares)
        testname = 'CARS_UNIF_0.2'
        onames = ['CARS_Unif_B1_p0.2']

    elif exp_name == 'CARS_sample1': # can add the date
        testname = 'CARS_test'
        onames = ['CARS_dir_name']


    else:
        print("ERROR")
        onames = []
        testname=''
        subdirnames=''
    
    dirname = join(os.path.dirname(__file__), testname)
    testdir = dirname + '/' + subdiropt
    subdirnames = [join(subdiropt, d) for d in listdir(testdir) if isdir(join(testdir, d))]
    if len(subdirnames) == 0:
        subdirnames = [subdiropt]
        
    subdirnames = [name+'/' for name in subdirnames]
    return onames, dirname, testname, subdirnames