import numpy as np
from matplotlib import pyplot as plt
import argparse

def plot_res(fname, Otype):
    err_gnorm = np.load(fname)
    errs = err_gnorm[0,:]
    gnorms = err_gnorm[1,:]
    plt.plot(errs)
    plt.yscale('log')
    plt.title(f'{Otype}: f(x) values (log scale) vs #queries')
    plt.show()
    plt.plot(gnorms)
    plt.yscale('log')
    plt.title(f'{Otype}: |grad(f)(x)| vs #queries')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-n", "--name", type=str, default="CARS_err_gnorm_params_r0.01_M10_bgt10k_exp_1_CHNROSNB.npy")
    args = vars(parser.parse_args())
    fname = "res/npy/"+args['name']
    Otype = args['name'].split('_')[0]
    plot_res(fname, Otype)