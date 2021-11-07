#!/usr/bin/env python
# coding: utf-8

# ## MNIST Attack
# This script loads the pre-trained model and attacks the images.
# Sample Usage: python MNIST_Attack.py -o CARS -name "CARS_dir_name" -v 1 -eps 0.3 -si 0.3 -t 0 10
# 
# DEFINITION of TID (testset id):
#   Label "n" in TID "t" means it is the "t"-th "n" appearing in the test set.
#   And this image(as a 1-D np array) is stored in "atk_test_set[t][n]"
#   So, for instance, "atk_test_set[0]" contains 10 images,
#     which are the first 0, first 1, ..., first 9 in the test set.
# 
# @author: (anonymous)
# 

# In[1]:


# basics
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from Network import Net
import utils
from datetime import datetime
import os #to make dir
import argparse
import random

# import our optimizers/helpers
from DataLoaders import MNIST_Loaders
import AttackTools
import TestTools
import ReadData as rd

# In[2]:
'''
sample example to run this code:
    python MNIST_Attack.py -o CARS -name "CARS_dir_name" -v 1 -eps 0.3 -si 0.3 -t 0 10
'''
if __name__ == "__main__":
    '''
    setting from the user input
    '''
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-o", "--oname", type=str, default="CARS",
                        help = "Allowed inputs: CARS, SQUARE, STP, SMTP, NS, SPSA, SPSA2")
    parser.add_argument("-t", "--tid", nargs = '+', type=int,
                        help = "Starting/Final Test set ID (TID). Between 0 and 1135.\nStarting TID must be given. Final is optional") 
    parser.add_argument("-b", "--budget", type = int, default = 10000,
                        help = "Budget for the function queries ")
    parser.add_argument("-eps", "--epsilon", type = float, default = 0.2,
                        help = "L-infty perturbation limit, between 0 and 1 ")
    parser.add_argument("-sr", "--samp_rad", type = float, default = 1.,
                        help = "Sampling radius")
    parser.add_argument("-v", "--verbal", type=int, default=1,
                        help = "Verbal option (0, 1, 2, 3): larger means more descriptive")
    parser.add_argument("-name", "--name_to_save", type=str, default="",
                        help = "name of the optimizer (used for save)")
    parser.add_argument("-nq","--NumQuad", type=int, default=0,
                        help = "Use numerical quadrature for CARS with NumQuad points.\
                            Set to 0 if not using NQ but Central Difference")
    parser.add_argument("-si","--shift_init", type=float, default=0.2,
                        help = "Shift the initial point to the center of the Box (set of feasible imgs).\
                            It should be less than eps.")
    parser.add_argument("-df","--display_freq", type=int, default=10,
                        help = "Show the success rate, average/median #queries every [display_freq] tid's.")
    parser.add_argument("-wsp","--window_size_param", type=float, default=0.2,
                        help = "Window size parameter for Square distribution. win sz = sqrt(p*width*height)")
    parser.add_argument("-dd", "--dist_dir", type=str, default="unif",
                        help = "Distribution of random directions. Default = unif. Options: square (square vectors), coord (random coordinate basis, unif)")

    args = vars(parser.parse_args())
    print(args)

    oname = args['oname']
    dist_dir = "square" if oname in ["CARS", "SQUARE"] else args['dist_dir'] # square for CARS
    tids = args['tid'] # testset id (from - to). More than 2 inputs --> ignored
    tid_from = tids[0]
    if len(tids)>1:
        tid_to = tids[1]
    else:
        tid_to = tid_from + 1
    budget = args['budget'] # budget for function queries
    epsilon = args['epsilon'] # L-infty norm tolerance for distortion
    r = args['samp_rad'] # sampling radius
    verbal = args['verbal'] # the larger, the more descriptive
    name_to_save = args['name_to_save']
    nq = args['NumQuad']
    if len(name_to_save)==0: # default is
        name_to_save = oname # the optimizer name
    ishift = args['shift_init']
    dsp_freq = args['display_freq']
    wsp = args['window_size_param']
    if epsilon<ishift:
        raise Exception(f"shift_init ({ishift}) must be less than epsilon (distortion limit, {epsilon})")
    

    '''
    Load the pre-trained model, and the test data set
    '''
    # Define the data loader, which fetchers batch_size number of images and stores them
    # in a tensor.
    batch_size = 64
    train_loader, test_loader = MNIST_Loaders(batch_size)

    # Fetch the pretrained model
    model = Net()
    state = torch.load('models/MNIST_weights.pth')
    model.load_state_dict(state['net_state_dict'])

    # Test the network accuracy on the test data set. It should be > 99%
    test_loss, test_acc, correct = utils.get_stats(model, test_loader)
    print('Test loss = {:.3f} and Test accuracy = {:.3f}'.format(test_loss, test_acc))

    # Set up the test set for an attack
    atk_test_set = [{} for _ in range(1135)] # "1" has 1135 samples (max) (cf. "5" has 892 samples (min))
    testset_counts = [0 for _ in range(10)]
    for dtest_, labels_ in test_loader:
        for i in range(dtest_.size()[0]):
            idx = labels_[i].item()
            atk_test_set[testset_counts[idx]][idx] = dtest_[i,:,:,:].flatten().numpy() # 1-D np array
            testset_counts[idx] += 1
    # each test set ( atk_test_set[tid] ) has at most one image for each label
    # When tid>892, it has missing labels (i.e., len(atk_test_set[tid]) < 10)

    print(testset_counts) # (number of images with label i) = testset_counts[i] at this point

    '''
    Parameter Setup for "attack" tool
    '''
    # attack tool setup first
    atk = AttackTools.AttackTool(model, c=2., kappa=20., eps = epsilon, metric = 'inf') # to compare with Zhao et al 2020
    # c and kappa used for C-W model. 
    #  *** For L-infty constraint model, c and kappa are not used ***
    
    # options for the data/attack
    options = { 'normalize': False, # determines whether the data will be re-normalized (so that they're in [0,1]^n)
                'metric': 'inf', # 'inf' for L-inf perturbation limit
                'constrained': True # imgs constrained in [0,1]^n
            }

    # parameter setting (as given in the parsed arguments and the options above)
    param = {'dim': 28*28, # MNIST img size
             'r': r, # sampling radius
             'budget': budget,
             'atk':atk, # an object containing tools for adversarial attack
             'constrained': options['constrained'], 
             'nq': nq,
             'ishift': ishift,
             'wsp':wsp,
             'dist_dir':dist_dir
            }
    opt = TestTools.opt_setup(Otype=oname, param=param)

    # opts (dict of optimizers) can be used to compare multiple optimizers at once
    opts = {name_to_save: opt}


    '''
    Set the test data, and the save directory before attack
    '''
    # Choose the data and set the save directory
    if tid_from < 1135:
        TEST_IDS = [i for i in range(tid_from,tid_to)] # set test data to attack
    else: # tid_from > 1135, it will choose "tid_to" number of random samples 
        TEST_IDS = random.sample(range(0,1135), tid_to) # randomly choose a part of data
    savedir = f'Results_{name_to_save}_{tid_from}' # directory to save the results
    if not os.path.exists(savedir):
                os.makedirs(savedir)
    with open( savedir+'/param.csv', 'w') as file:
        file.write(f'optimizer,{oname}\n')
        file.write(f'name,{name_to_save}\n')
        for key in param:
            if key != 'atk':
                file.write(f'{key},{param[key]}\n')
            else:
                file.write(f'atk.c,{atk.c}\n')
                file.write(f'atk.kappa,{atk.kappa}\n')
                file.write(f'atk.eps,{atk.eps}\n')
                file.write(f"normalize,{options['normalize']}\n")

    currtime = datetime.now()
    print(currtime)

    '''
    Attack starts here
    '''
    for testset_id in TEST_IDS: # for each TID,
        print(f'tid = {testset_id}') # printed even when verbal = 0
        T = {} # Tester class objects
        T[testset_id] = TestTools.Tester(atk_test_set = atk_test_set[testset_id], atk = atk,
                            opts = opts, options = options, tid=testset_id)

        lbl_filter = [i for i in range(10)] # do attack only when the label is in this set (default: attack all labels)

        # attack
        for lbl in atk_test_set[testset_id]:
            if lbl in lbl_filter:
                T[testset_id].run_single_test(lbl, target_lbl = None, verbal = verbal)
        prevtime = currtime
        currtime = datetime.now()
        print(f'{currtime}\telpased time:{currtime-prevtime}')

        # Show/Save results
        for lbl in atk_test_set[testset_id]:
            if lbl in lbl_filter:
                # display_single_res is disabled by default. Use this to see the (visual) results immediately
                # T[testset_id].display_single_res(lbl, opts, cmap='gray', save = f'Res/testset={testset_id}_Linf=0.2_Lbl={lbl}.png', show = False) # omit cmap, or set it None for color images
                T[testset_id].save_res_simple(testset_id, lbl, opts, subdir=savedir)
        if verbal>0:
            if (testset_id+1) % dsp_freq == 0:
                tids = [tid_from]
                rd.show_res(name_to_save, tids, name_to_save)




