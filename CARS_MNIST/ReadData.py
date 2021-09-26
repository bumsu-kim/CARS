

import argparse
import csv
import numpy as np

'''
    Sample usage:
    (1) python ReadData.py -o CARS_test -t 0
        if the result file to analyze is
            /Results_CARS_test_0/CARS_test.csv

    (2) python ReadData.py -n TEST01 -o CARS_test -t 0 350 700    
        if the result files to analyze are
            /Results_TEST01_0/CARS_test.csv
            /Results_TEST01_350/CARS_test.csv
            /Results_TEST01_700/CARS_test.csv
'''

def show_res(name, tids, oname):
    dat = np.empty([10000,5])
    i=0
    for tid in tids:
        fname = f'Results_{name}_{tid}/{oname}.csv'
        with open(fname, 'r') as f:
            reader = csv.reader(f, dialect='excel', delimiter = '\t')
            for row in reader:
                dat[i,0:4] = [int(str) for str in row[:-1]]
                if row[-1] == 'S':
                    dat[i,-1] = 1
                else:
                    dat[i,-1] = 0
                i+=1 # count the total #rows
    dat = dat[:i, :]
    dat = dat.astype('int')
    idx= dat[:,-1]>0
    succ = np.sum(idx)
    succ_rate = succ/len(dat)
    avg = np.average(dat[idx,3])
    med = np.median(dat[idx, 3])
    print(f'{oname} (#imgs = {len(dat)}): \n'
            f'\tsuccess rate    = {succ_rate}\n'
            f'\taverage queries = {avg}\n'
            f'\tmedian queries  = {med}' )




if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-o", "--oname", type=str, default="CARS")
    parser.add_argument("-t", "--tid",nargs = '*', default=None)
    parser.add_argument("-n", "--name", type=str, default="")
    args = vars(parser.parse_args())
    print(args)
    oname = args['oname']
    if len(args['name'])==0:
        name = oname
    else:
        name = args['name']
    tids = args['tid']
    show_res(name, tids, oname)
    