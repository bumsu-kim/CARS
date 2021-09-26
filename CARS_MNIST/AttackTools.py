import numpy as np
import torch
import torch.nn.functional as F


class AttackTool:

    def __init__(self, model,
                 data = None,
                 label = None, 
                 kappa = 20.0,
                 c = 20.0,
                 eps = 0.2,
                 target_lbl=None,
                 metric = None):
        self.model = model
        self.data = data # valid image data (pxl val in [0, 1])
        self.label = label # label of "data"
        self.target_lbl = target_lbl
        # MNIST image size (1 img x 1 channel x Height x Width)
        self.viewsize = (1, 1, 28, 28)
        self.kappa = kappa #20.0 by default
        self.c = c #20.0 by default
        self.metric = metric
        self.constrained = False
        self.eps = eps
        self.ub = 1. # upper bound
        self.lb = 0. # lower bound

    def setdata(self, data):
        self.data = data
    def setlabel(self, label):
        self.label = label
    def settargetlabel(self, tlabel):
        self.target_lbl = tlabel
    def xmap(self, x):
        if self.constrained:
            return x
        else:
            return np.tanh(x)*0.51 + 0.5 # transform to a valid img (pxl val in [0, 1])
    def xmap_inv(self, y):
        if self.constrained:
            return y
        else:
            return np.arctanh( (y-0.5)/0.51 ) # inverse transform of xmap
    # assume x is a valid image
    def logp_over_classes(self, x, out=None, viewsize = None):
        if viewsize == None:
            viewsize = self.viewsize
            
        if x.dtype != 'float32':
            x = x.astype('f')
        if out == None:
            return self.model(torch.tensor(x).view(viewsize)).detach().numpy()
        else:
            return out.detach().numpy()

    # assume x is a valid image
    def p_over_classes(self, x, out = None, viewsize = None):
        if viewsize == None:
            viewsize = self.viewsize
            
        if x.dtype != 'float32':
            x = x.astype('f')
        if out == None:
            return np.exp(self.model(torch.tensor(x).view(viewsize)).detach().numpy())
        else:
            return [np.exp(out_i.detach().numpy()) for out_i in out]
    
    # assume x is a valid image
    def nll(self, x, lbl, out=None):
        if x.dtype != 'float32':
            x = x.astype('f')
        if out == None:
            return F.nll_loss(self.model(torch.tensor(x).view(self.viewsize)), torch.tensor([lbl])).item()
        else:
            return F.nll_loss(out, torch.tensor([lbl])).item()

    # assume x is a valid image
    def CW(self, x, out = None):
        if x.dtype != 'float32':
            x = x.astype('f')
        if out == None:
            out = self.model(torch.tensor(x).view(self.viewsize))
        if self.target_lbl == None: # untargeted attack
            label = self.label
            nll = self.nll(x, label, out)
            others = [self.nll(x, lbl, out)
                    for lbl in range(len(out[0])) if lbl != label]
            CWval = max(-self.kappa, min(others)-nll)
        else: # targeted attack
            label = self.target_lbl
            nll = self.nll(x, label, out)
            others = [self.nll(x, lbl, out)
                    for lbl in range(len(out[0])) if lbl != label]
            CWval = max(-self.kappa, nll - min(others))
        return CWval
    
    # assume x is a valid image
    def curr_lbl(self, x = None, pdist = None): # curently classified label
        if pdist == None:
            return np.argmax(self.p_over_classes(x))
        else:
            return np.argmax(pdist)

    # assume x is a valid image
    def __call__(self, x): # returns true if the attack was successful
        if x.dtype != 'float32':
            x = x.astype('f')
        out = self.model(torch.tensor(x).view(self.viewsize))
        fval = self.overall_loss(x, out = out)
        pdist = self.p_over_classes(x, out = out)
        label = self.curr_lbl(pdist = pdist)
        returned = {'fval': fval, 'pdist': pdist, 'label': label}
        # also determine if the attack was successful
        if self.target_lbl == None:
            returned['atk_succ'] = (returned['label'] != self.label)
        else:
            returned['atk_succ'] = (returned['label'] == self.target_lbl)

        return returned

    def dist_from_orig(self, x):
        if self.metric == None: # default is L2
            return np.linalg.norm(x-self.data)**2
        elif self.metric == 'L1' or self.metric == 1:
            return np.sum( np.abs(x-self.data) )
        elif self.metric == 'inf' or self.metric == float("inf"):
            return np.max( np.abs(x-self.data) )
    
    def proj(self, x, metric = 'inf'):
        if self.metric == 'inf':
            delta = x-self.data
            delta[delta>self.eps] = self.eps*0.999
            delta[delta<-self.eps] = -self.eps*0.999
            y = self.data + delta
            y[y>1.] = 1.
            y[y<0.] = 0.
        return y
    
    # assume x is a valid image
    def overall_loss(self, x, out = None):
        if not self.constrained:
            return  self.CW(x, out=out)*self.c + \
                    self.dist_from_orig(x)
        else:
            return self.CW(x, out=out)
    
