import numpy as np
from matplotlib import pyplot as plt
import argparse

def plot_res(fname, Otype, pname, expnum = None):
    err_gnorm = np.load(fname)
    errs = err_gnorm[0,:]
    gnorms = err_gnorm[1,:]
    if expnum is not None:
        expnumtxt = f"{expnum}"
    else:
        expnumtxt = ""
    #plt.rc('text', usetex = True)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 5))
    fmin = np.min(errs) - 1e-30
    ax1.plot(errs - fmin )
    ax1.set_yscale('log')
    ax1.title.set_text(f'{Otype}: [{expnumtxt}] {pname} f-f* vs #queries')


    ax2.plot(gnorms)
    ax2.set_yscale('log')
    ax2.title.set_text(f'{Otype}: [{expnumtxt}] |grad({pname})| vs #queries')

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    
    # old version
    parser.add_argument("-n", "--name", type=str, default="CARS_err_gnorm_params_r0.01_M10_bgt10k_exp_1_CHNROSNB.npy")
    args = vars(parser.parse_args())
    fname = "res/npy/"+args['name']
    strs = args['name'].split('_')
    Otype = strs[0]
    pname = strs[-1].split('.')[0]
    exp_num = strs[-2]
    plot_res(fname, Otype, pname, exp_num)

    # # new version
    # parser.add_argument("-o", "--otype", type=str, default="CARS")
    # parser.add_argument("-par", "--params", type=str, default="r0.01_M10_bgt10k")
    # parser.add_argument("-prob", "--problem_name", type=str, default="CHNROSNB")
    # parser.add_argument("-e", "--exp_num", type=str, default="0")
    # args = vars(parser.parse_args())
    # Otype = args['Otype']
    # pname = args['problem_name']
    # exp_num = args['exp_num']
    # fname = f"res/npy/{args['params']}/{Otype}_{pname}_{exp_num}.npy"

    # plot_res(fname, Otype, pname, exp_num)