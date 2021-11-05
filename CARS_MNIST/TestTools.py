import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt 
from DataLoaders import MNIST_Loaders
from Network import Net
import utils
import AttackTools
import optimizers
import copy
import os #to make dir

def opt_setup(Otype, param):
    if 'r' not in param:
            param['r'] = 1.
    if 'budget' not in param:
        param['budget'] = 10000
    if 'target_fval' not in param:
        param['target_fval'] = None
    if 'constrained' not in param:
        param['constrained'] = False
    if 'metric' not in param:
        param['metric'] = 'inf'
        
    if Otype.upper() == 'RING':
        if 'useFIM' not in param:
            param['useFIM'] = False
        if 'm' not in param:
            param['m'] = 8
        if 'num_blocks' not in param:
            param['num_blocks'] = 49*2
        if 'BweightOption' not in param:
            param['BweightOption'] = None
        param['useFIM'] = True
        opt = optimizers.RING(param, y0 = None,f = None)
    elif Otype.upper() == 'SHIPS':
        param['useFIM'] = False
        opt = optimizers.RING(param, y0 = None,f = None) # param['useFIM'] = False
    elif Otype.upper() == 'CARS' or Otype.upper() == 'ZOO':
        opt = optimizers.CARS(param, y0 = None,f = None)
    elif Otype.upper() == 'SQUARE':
        opt = optimizers.SquareATK(param, y0 = None, f = None)
    elif Otype.upper() == 'STP':
        opt = optimizers.STP(param, y0=None, f=None)
    elif Otype.upper() == 'SMTP':
        opt = optimizers.SMTP(param, y0=None, f=None)
    elif Otype.upper() == 'NS':
        opt = optimizers.NS(param, y0=None, f=None)
    return opt

def vec2Img(vec):
    ImageSize = [28, 28] #image size for MNIST
    return vec.reshape(ImageSize)

class result:
    def __init__(self):
        self.name = None # optimizer's name
        self.isol = None # initial sol
        self.ilbl = None # initial label
        self.tsol = None # terminal sol
        self.tlbl = None # terminal label
        self.fevals = None # function evaluations
        self.niter = None # number of iterations
        self.status = None # status after test
        self.performance_log = None # performance log
        self.CARScounter = None # if available
    
    def saveResult(self, name, opt, performance_log = None):
        self.name = name
        self.isol = opt.Atk.data
        self.ilbl = opt.Atk.label
        self.tsol = opt.ximg
        self.tlbl = opt.Atk.curr_lbl(self.tsol)
        self.fevals = opt.function_evals
        self.niter = opt.t
        self.status = opt.status
        self.performance_log = performance_log
        if hasattr(opt, 'CARScounter'):
            self.CARScounter = opt.CARScounter

    def showBriefResult(self):
        if hasattr(self, 'CARScounter'):
            print(f'opt = {self.name}, niter = {self.niter}, CARScounter = {self.CARScounter}')
            print(f'\tfunction evaluations = {self.fevals}, status = {self.status}')
            print(f'\tori lbl = {self.ilbl}, curr lbl = {self.tlbl}')

    def plotResult(self, cmap = None):
        # assume a proper subplot is already set
        if cmap == None:
            plt.imshow(vec2Img(self.tsol))
        else:
            plt.imshow(vec2Img(self.tsol), cmap = cmap)
        plt.title(f'{self.tlbl}')
        # plt.title(f'lbl ({self.name}) = {self.tlbl}')
    
    def save2file(self, f, tid, delim = '\t'):
        '''
        saves only 
        opt name, testset_id, orig label, final label, num_evals
        file name:
        (opt_name).csv
        content:
        each row consists of:
        [testset_id, orig_label, final_label, num_evals, status]
        here status = 'B'(budget reached), 'T'(target reached), or 'S'(attack succeeded)

        @param
        f ..... file object (open, append mode)
        t ..... testset_id
        '''
        f.write(f'{tid}{delim}{self.ilbl}{delim}{self.tlbl}{delim}{self.fevals}{delim}{self.status}\n')



class Tester:
    '''
    A test suite, containing the set of test data and set of optimizers
    Usage:
    1. Set data
    2. Set optimizers
    3. Run .run_single_test(label, target_label)
    '''
    def __init__(self, atk_test_set, atk, opts = None, options = None, tid = 0):
        """
        (inputs)
        atk_test_set .. dict
                        (*this)[label] = image data in a vector, whose label is "label"
        atk ........... AttackTools object
        opts .......... dict
                        a dict of optimizers to test
                        key = optimizer's name (possibly with options)
        options ....... dict
                        ['normalize'] --> images will be normalized to have pixel values btw 0 and 1

        (attributes)
        res ........... dict of dict of result class
                        res[optname][lbl] = Attack Results containing
                        - initial / terminal solutions
                        - initial / terminal labels
                        - function_evals
                        - number of iterations
                        - status
                        - CARScounter (if available)
                        - performance log
        """
        self.atk = atk
        self.atk_test_set = atk_test_set
        if opts != None:
            self.setopts(opts)

        self.tid = tid
        if 'normalize' in options:
            if options['normalize']:
                self.normalize_data()
        if 'metric' in options:
            self.metric = options['metric']
        if 'constrained' in options:
            self.constrained = options['constrained']

    def setopts(self, opts):
        self.opts = opts
        self.res = {} # will be a dict(key: opt) of dict(key: labels) of status
        for optname in opts:
            self.res[optname] = {}

    def addopts(self, added):
        self.opts = {**self.opts, **added} # merge two dictionaries
        for optname in added:
            self.res[optname] = {}

    def normalize_data(self):
        # re-normalize the images to have pixel values in [0, 1]
        max_pixel_val = max([ np.max(self.atk_test_set[lbl]) for lbl in self.atk_test_set])
        min_pixel_val = min([ np.min(self.atk_test_set[lbl]) for lbl in self.atk_test_set])
        for lbl in self.atk_test_set:
            self.atk_test_set[lbl] = (self.atk_test_set[lbl]-min_pixel_val)/(max_pixel_val-min_pixel_val)

    def run_single_test(self, label, selected_opts = None, target_lbl = None, verbal = 2):
        # verbal option: 0 --> no output
        #                1 --> output when started/finished
        #                2 --> 1 + func vals
        if label not in self.atk_test_set:
            print(f'{label} is not a valid label in this attak test set.')
        self.atk.setdata(self.atk_test_set[label])
        self.atk.setlabel(label)
        self.atk.target_lbl = target_lbl
        self.atk.metric = self.metric
        self.atk.constrained = self.constrained
        # self.atk.settargetlabel(target_lbl)
        if selected_opts == None:
            opts = self.opts
        else:
            opts = selected_opts
        for oname in opts:
            self.res[oname][label] = result()
            # setup
            opt = copy.deepcopy(self.opts[oname]) # to reset everytime
            # otherwise the shallow copied opt may alter the original opt object
            opt.setAtkAll(  Atk = self.atk,
                            y0  = self.atk_test_set[label],
                            f   = lambda x:self.atk(x)     )
            performance_log = []
            status = None
            # verbal
            if verbal > 0:
                if target_lbl != None:
                    print(f'start atd atk ({oname}) on lbl = {label}, target lbl = {self.atk.target_lbl}')
                else:
                    print(f'\t[{label}]', end='\t')#start an untargeted atk ({oname}) on lbl = {label}')
            # actual attack starts here
            while status == None: 
                # one iteration
                evals, _xfinal, status = opt.step()
                # logging
                performance_log.append( [evals, opt.fval])
                # print
                if verbal > 2:
                    if opt.t % int(10*np.log(opt.t+10)) == 0:
                        opt.report('f(x): %f  F-evals: %d\n' %
                        (opt.fval, evals) )
            # logging
            self.res[oname][label].saveResult(oname, opt, performance_log)
            if verbal>1:
                if opt.t>0:
                    if opt.Otype in ['CARS']:
                        print(f"CARS: {opt.CARScounter},\tCVX = {opt.cvx_counter/opt.t*100:.1f} %", end='\t')
                # print(f"CVX counter: {opt.cvx_counter}")
                # print(f"Final distortion: {opt.Atk.dist_from_orig(opt.x)}")
                np.sum((_xfinal-opt.xinit)**2)
                print(f"distortion (L2) = { np.sum((_xfinal-opt.xinit)**2) }", end = '\t')
                print(f"(Linf) = {np.amax(_xfinal-opt.xinit):.2f}", end='\t')
            if verbal > 0:
                print( f"#iter = {opt.t} (#eval = {evals}),\t final status = {status}")


    def display_single_res(self, label, opts_names_to_disp = None, cmap = None, title = None,
            onlyImg = False, onlyLoss = False, save = None, show = False, logplot = True, savedir = None):
        if title == None:
            title = 'RING for MNIST ATK'
        if onlyImg == False and onlyLoss == False:
            plt.subplot(2,1,1)
            plt.cla()
            legends = []
            if opts_names_to_disp == None: # default: display all
                opts_names_to_disp = self.opts # only need the names (keys of the dict)
            for oname in opts_names_to_disp:
                if logplot:
                    plt.plot(np.array(self.res[oname][label].performance_log)[:,0],
                    np.log10(np.array(self.res[oname][label].performance_log)[:,1]), linewidth=1, label = oname)
                else:
                    plt.plot(np.array(self.res[oname][label].performance_log)[:,0],
                (np.array(self.res[oname][label].performance_log)[:,1]), linewidth=1, label = oname)
                legends.append(oname)
            plt.title(title)
            plt.xlabel('function evaluations')
            plt.ylabel('$log($ f $)$')
            plt.legend(legends)
            
            nopts = len(opts_names_to_disp) 
            plotnum = 1 
            # show original image
            plt.subplot(2,nopts+1,nopts+2)
            if cmap == None:
                plt.imshow(vec2Img(self.atk_test_set[label]))
            else:
                plt.imshow(vec2Img(self.atk_test_set[label]), cmap = cmap)
            plt.title(f'original label = {label}')
            # show attacked
            for oname in opts_names_to_disp:
                plt.subplot(2, nopts+1, nopts+2 + plotnum)
                self.res[oname][label].showBriefResult()
                self.res[oname][label].plotResult(cmap)
                plotnum += 1
            plt.tight_layout(pad = 1.0)
            if show:
                plt.show()
        elif onlyImg == True: # plot only images
            nopts = len(opts_names_to_disp) 
            # show original image
            # set number of rows
            if nopts < 8:
                nr = 2
            elif nopts < 12:
                nr = 3
            else:
                nr = 4
            nc = int(np.ceil((nopts+1)/nr))
            plt.subplot(nr,nc,1)
            if cmap == None:
                plt.imshow(vec2Img(self.atk_test_set[label]))
            else:
                plt.imshow(vec2Img(self.atk_test_set[label]), cmap = cmap)
            plt.title(f'original label = {label}')
            # show attacked
            plotnum = 2
            for oname in opts_names_to_disp:
                plt.subplot(nr, nc, plotnum)
                self.res[oname][label].showBriefResult()
                self.res[oname][label].plotResult(cmap)
                plotnum += 1
            plt.tight_layout(pad = 1.0)
            if show:
                plt.show()
        elif onlyLoss == True: # plot only loss
            legends = []
            if opts_names_to_disp == None: # default: display all
                opts_names_to_disp = self.opts # only need the names (keys of the dict)
            for oname in opts_names_to_disp:
                plt.plot(np.array(self.res[oname][label].performance_log)[:,0],
                np.log10(np.array(self.res[oname][label].performance_log)[:,1]), linewidth=1, label = oname)
                legends.append(oname)
            plt.title(title)
            plt.xlabel('function evaluations')
            plt.ylabel('$log($ f $)$')
            plt.legend(legends)
            if show:
                plt.show()
        if save != None:
            if savedir != None:
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                save = savedir+save # add directory to the file name
            plt.savefig(save)
    
    def save_res_simple(self, testset_id, label, opts_names_to_save = None, subdir = 'Res'):
        '''
        results are saved in the subdir folder (the folder will be created if not exsits)
        1. saves the brief result
        file name: (opt_name).csv
        content: each row consists of:
            [testset_id, orig_label, final_label, num_evals]
        2. also saves the original/attacked images as 28*28 numpy array
        original img name: (label)_(testset_id).npy
        attacked img name: (label)_(testset_id)_(opt_name)_(final_label).npy
        '''
        if opts_names_to_save == None: # default: save all
                opts_names_to_save = self.opts # only need the names (keys of the dict)
            
        for oname in opts_names_to_save:
            save_orig = True
            res = self.res[oname][label]
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            fname = subdir + f'/{oname}.csv'
            f = open(fname, 'a')
            res.save2file(f=f, tid = testset_id, delim='\t')
            f.close()
            if save_orig:
                subdir_img = 'img_'+ subdir
                if not os.path.exists(subdir_img):
                    os.makedirs(subdir_img)
                fname = subdir_img + f'/{label}_{testset_id}.npy'
                np.save( fname, vec2Img(res.isol))
                save_orig = False
            fname = subdir_img + f'/{label}_{testset_id}_{oname}_{res.tlbl}.npy'
            np.save( fname, vec2Img(res.tsol))

            



