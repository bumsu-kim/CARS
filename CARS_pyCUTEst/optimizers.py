import numpy as np
import numpy.matlib
from optbase import BaseOptimizer
from multiprocessing.dummy import Pool
import OptTools as ot
import copy
from matplotlib import pyplot as plt 

class OptForAttack(BaseOptimizer):
    '''
    Base class for RING, SHIPS, CARS and other optimizers for black box adversarial attacks
    Used for common parameter settings
    '''
    def __init__(self, param, y0, f):
        '''
            param ..... dict
                        contains hyperparameters (given as options) and
                         important objects
                        - atk       : contains the tools for adversarial attack
            y0 ........ np array
                        original image in a 1-D array form
            f ......... function handle
                        Loss functino to minimize (see the appendix of the paper)
        '''
        
        self.n = param['dim'] # problem dimension, 28*28 for MNIST imgs

        
        
        self.stationarity_threshold = param['threshold_first_order_opt']

        self.function_budget = int(param['budget'])
        self.function_target = param['target_fval']
        self.r = param['r'] # initial sampling radius

        self.fmin = float('inf')
        self.xmin = None

        self.nq = param['nq'] # numerical quadrature points
        
        # record where the min f value came from. (counts [x0, x +- ru, x_{CARS}])
        self.CARScounter = numpy.zeros(3, dtype=int)
        self.alpha = 0.5 # step size param = 1/Lhat

        # Initialize values
        self.function_evals = 0 # Initialize the number of function evaluations
        self.t = 0 # iteration counter
        self.status = None # stop if 'B'(udget), 'T'(arget), 'S'(uccessful attack)
        self.cvx_counter = 0 # count the convex (f'' along u > 0) iter in CARS()

        # distribution of random directions (default = sq (square))
        self.dist_dir = param['dist_dir']
        if self.dist_dir.upper() == 'SQUARE':
            #self.rtype = 'Box'
            pass # not supported
        elif self.dist_dir.upper() in ['UNIF', 'NORMAL', 'GAUSSIAN']:
            self.rtype = 'Uniform'
        elif self.dist_dir.upper() == 'COORD':
            self.rtype = 'Coord'
        else: # raise exception
            raise Exception(f"dist_dir ({self.dist_dir}) must be one of 'square', 'unif', 'normal', or 'coord'")
        
        self.sety0(y0)
        
        self.setf(f)

    ''' 
    Setters
    '''

    def sety0(self, y0): # Set up the initial point
        self.xinit = np.copy(y0)
        self.x = np.copy(y0)

    def setf(self, f):
        self.f = lambda x: self.eval(f, x, record_min = True)
        self.fval = f(self.x)
        self.fmin = self.fval
        self.f_norecording = lambda x: self.eval(f, x, record_min = False)
        self.grad = lambda x: f(x, gradient=True)[1] # does not count as a func eval, nor record the min
        self.fvalseq = np.zeros(self.function_budget+1)
        self.fvalseq[0] = self.fval

    '''
    function evaluation
    '''
    def eval(self, f, x, record_min = True):
        ''' function evaluation at x
        '''
        self.curr_x = x
        res = f(self.curr_x)
        self.curr_grad_norm = np.linalg.norm(self.f(x, gradient=True)[1])

        # record the minimum f
        if record_min:
            self.function_evals += 1 # count func evals
            if  res < self.fmin: # new min found
                self.xmin = x
                self.fmin = res
            self.fvalseq[self.t+1] = res # record as a f-value

        return res

    
    

    '''
    Stopping Condition Check
    'B' -- Max budget reached
    'T' -- Target value reached
    'S' -- Stationary (reached First Order Optimality)
    '''
    def stopiter(self):
        #self.curr_grad_norm = np.linalg.norm(self.grad(self.x))
        # max budget reached
        if self.reachedFunctionBudget(self.function_budget, self.function_evals):
            self.status = 'B'

        # target reached
        if self.function_target != None:
            if self.reachedFunctionTarget(self.function_target, self.fval):
                self.status = 'T'
                
        if self.reachedFirstOrderOptimality(self.curr_grad_norm, self.stationarity_threshold):
            self.status = 'S'
            

    def CARS_step(self, u, r):
        ''' Curvature Aware (Random) Search Step
        here, we assume the direction u is given.
        d = f',  h= f''
        (' denotes the directional derivative along u)

        ** update rule: x_{CARS} = x - alpha * d/h u  , when h > 0 **

        Investigates 3 or 4 pts, depending on the sign of h:
            f(x+ru), f(x-ru), f(x-t_{bd}*d*u), and f(x_{CARS})
                where x-t_{bd}*u is on the boundary, i.e.,
                    t_{bd} = max { t>0: x-t*d*u is in the feasible set }

        (input)
        u ......... Direction
        r ......... sampling radius

        (output)
        fmin .... minimum f value
        xmin .... argmin of f
        '''
        
        # alpha = 1/Lhat <= 1
        
        eps = 1e-6 # regularization for d2 (prevent overflow)
        if self.nq == 0: # regular CARS
            d, d2 = ot.CentDiff(self.f, self.x, r, u, self.fval, eps)
            Lhat = 1/2 + np.sqrt(1/4 + self.M*np.abs(d/d2**2/2))
            alpha = 1/Lhat
        else: # CARS-NQ
            d, d2, d3, d4 = ot.NumQuad(self.f, self.x, r, u, self.fval, self.Atk, self.nq)
            d3max = np.abs(d3) + np.abs(d/d2*d4) # estimate of sup|f'''| near x
            # 1/Lhat estimated from higher order derivatives
            alpha = self.alpha/ ( 0.5 + np.sqrt( 0.25 + np.abs(d*d3max/d2**2/3))) # see proposition 5.3

        #if d2 > 1e-8: # when convex
        xnew = self.x - alpha*d/d2*u
        self.cvx_counter += 1 # count this case
        fxnew = self.f(xnew)
        
        # check min
        if self.fmin == self.fval:
            minidx = 0 # not moved (xmin = x0)
        elif self.fmin == fxnew:
            minidx = 2
        else: # min is either x +- ru
            minidx = 1
        self.CARScounter[minidx] += 1
       
        return self.fmin , self.xmin
    

class CARS(OptForAttack):
    '''
    Curvature Aware Random Search
    '''
    def __init__(self, param, y0, f):
        '''
            Initialize parameters
        '''
        
        super().__init__(param, y0, f)
        self.Otype = 'CARS'
        self.M = param['M']
        ######### debug mode ########
        #print('shape of x:', np.shape(self.x))
        
        
    def sety0(self, y):
        super().sety0(y)

    def step(self, u = None):
        ''' 
            Do CARS Step.
            The direction vector u can be given as a param.
            If not given, it randomly generate a direction using the distribution parameter
                (-dd in script, self.rtype)
        '''
        r = self.r * np.sqrt(1/(self.t+1)) # decaying sampling radius (1/sqrt(k))
        if self.t==0:
            self.stopiter()
            if self.status != None:
                return self.function_evals, self.x, self.status
        # Take step of optimizer
        if u == None:
            # generate a random direction
            if self.rtype == 'Box':
                # u = ot.sampling( n_samp = 1, dim = self.n, randtype = self.rtype,
                #             distparam = {'coord': ot.idx2coord(np.random.randint(0, np.size(self.x))),
                #                 'windowsz': int(np.round(np.sqrt(np.prod(self.Atk.viewsize[2:4])*self.p))),
                #                 'ImSize': self.Atk.viewsize[2:4]
                #                 }
                #             )
                pass
            elif self.rtype == 'Uniform':
                u = ot.sampling(n_samp = 1, dim = self.n, randtype = self.rtype)
            elif self.rtype == 'Coord':
                u = ot.sampling(n_samp = 1, dim = self.n, randtype = self.dist_dir)

            # normalize
            u /= np.linalg.norm(u)
            
            
        u = u.reshape(np.shape(self.x))
        ######### debug mode ########
        #print('shape of u:', np.shape(u))
        fmin, xmin = self.CARS_step(u, r)
        self.x = xmin
        self.fval = fmin

        self.t += 1
        self.stopiter()
        
        if self.status != None:
            return self.x, self.fvalseq[0:self.t+1], self.function_evals, self.curr_grad_norm, self.status, True # 3rd val = termination or not
        else:
            return self.x, self.fvalseq[0:self.t+1], self.function_evals, self.curr_grad_norm, self.status, False # 3rd val = termination or not
        #else:
        #    return self.function_evals, None, None   


#######################################################################################
## NEED UPDATES 

class SMTP(OptForAttack):
    '''
    Stochastic Three Point method (with Momentum)
    '''
    def __init__(self, param, y0, f):
        '''
            Initialize parameters
        '''
        
        super().__init__(param, y0, f)
        if param['momentum']:
            self.Otype = 'SMTP'
        else:
            self.Otype = 'STP'

        self.M = 1
        ######### debug mode ########
        #print('shape of x:', np.shape(self.x))
        
        
    def sety0(self, y):
        super().sety0(y)

    def step(self, u = None):
        ''' 
            Do CARS Step.
            The direction vector u can be given as a param.
            If not given, it randomly generate a direction using the distribution parameter
                (-dd in script, self.rtype)
        '''
        r = self.r * np.sqrt(1/(self.t+1)) # decaying sampling radius (1/sqrt(k))
        if self.t==0:
            self.stopiter()
            if self.status != None:
                return self.function_evals, self.x, self.status
        # Take step of optimizer
        if u == None:
            # generate a random direction
            if self.rtype == 'Box':
                # u = ot.sampling( n_samp = 1, dim = self.n, randtype = self.rtype,
                #             distparam = {'coord': ot.idx2coord(np.random.randint(0, np.size(self.x))),
                #                 'windowsz': int(np.round(np.sqrt(np.prod(self.Atk.viewsize[2:4])*self.p))),
                #                 'ImSize': self.Atk.viewsize[2:4]
                #                 }
                #             )
                pass
            elif self.rtype == 'Uniform':
                u = ot.sampling(n_samp = 1, dim = self.n, randtype = self.rtype)
            elif self.rtype == 'Coord':
                u = ot.sampling(n_samp = 1, dim = self.n, randtype = self.dist_dir)

            # normalize
            u /= np.linalg.norm(u)
            
            
        u = u.reshape(np.shape(self.x))
        ######### debug mode ########
        #print('shape of u:', np.shape(u))
        fmin, xmin = self.CARS_step(u, r)
        self.x = xmin
        self.fval = fmin

        self.t += 1
        self.stopiter()
        
        if self.status != None:
            return self.x, self.fvalseq[0:self.t+1], self.function_evals, self.curr_grad_norm, self.status, True # 3rd val = termination or not
        else:
            return self.x, self.fvalseq[0:self.t+1], self.function_evals, self.curr_grad_norm, self.status, False # 3rd val = termination or not
        #else:
        #    return self.function_evals, None, None   

class SMTP(OptForAttack):
    '''
    SMTP
    '''
    def __init__(self, param, y0, f):
        '''
            Initialize parameters
        '''
        
        super().__init__(param, y0, f)
        self.Otype = 'CARS'
        
        
        ''' Other parameters
        p ...... Window size parameter. Fraction of pixels being changed
                 Thus the window size is sqrt(p)*28 for MNIST imgs
        '''
        self.p = self.wsp

    def sety0(self, y):
        super().sety0(y)

    def step(self, u = None):
        ''' 
            Do CARS Step.
            The direction vector u can be given as a param.
            If not given, it randomly generate a direction using the distribution parameter
                (-dd in script, self.rtype)
        '''

        if self.t==0:
            self.stopiter()
            if self.status != None:
                return self.function_evals, self.x, self.status

        if self.t>1:
            self.fminprev = self.fmin # previous min value
        # Take step of optimizer
        if u == None:
            # generate a random direction
            if self.rtype == 'Box':
                u = ot.sampling( n_samp = 1, dim = self.n, randtype = self.rtype,
                            distparam = {'coord': ot.idx2coord(np.random.randint(0, np.size(self.x))),
                                'windowsz': int(np.round(np.sqrt(np.prod(self.Atk.viewsize[2:4])*self.p))),
                                'ImSize': self.Atk.viewsize[2:4]
                                }
                            )
            elif self.rtype == 'Uniform':
                u = ot.sampling(n_samp = 1, dim = self.n, randtype = self.rtype)
            elif self.rtype == 'Coord':
                u = ot.sampling(n_samp = 1, dim = self.n, randtype = self.dist_dir)

            # normalize
            u /= np.linalg.norm(u)
        fmin, xmin = self.CARS_step(u, self.r)
        self.x = xmin
        self.ximg = self.Atk.xmap(self.x)
        self.fval = fmin

        self.t += 1
        self.stopiter()
        # decrease p as #iter increases
        if self.t in [2,10,40,250,500,800,1200,1600]:
            self.p /= 2.
        if self.status != None:
            return self.function_evals, self.x, self.status
        else:
            return self.function_evals, None, None   

class NS(OptForAttack):
    '''
    Curvature Aware Random Search
    '''
    def __init__(self, param, y0, f):
        '''
            Initialize parameters
        '''
        
        super().__init__(param, y0, f)
        self.Otype = 'CARS'
        
        
        ''' Other parameters
        p ...... Window size parameter. Fraction of pixels being changed
                 Thus the window size is sqrt(p)*28 for MNIST imgs
        '''
        self.p = self.wsp

    def sety0(self, y):
        super().sety0(y)

    def step(self, u = None):
        ''' 
            Do CARS Step.
            The direction vector u can be given as a param.
            If not given, it randomly generate a direction using the distribution parameter
                (-dd in script, self.rtype)
        '''
        if self.t==0:
            self.stopiter()
            if self.status != None:
                return self.function_evals, self.x, self.status
        # Take step of optimizer
        if u == None:
            # generate a random direction
            if self.rtype == 'Box':
                u = ot.sampling( n_samp = 1, dim = self.n, randtype = self.rtype,
                            distparam = {'coord': ot.idx2coord(np.random.randint(0, np.size(self.x))),
                                'windowsz': int(np.round(np.sqrt(np.prod(self.Atk.viewsize[2:4])*self.p))),
                                'ImSize': self.Atk.viewsize[2:4]
                                }
                            )
            elif self.rtype == 'Uniform':
                u = ot.sampling(n_samp = 1, dim = self.n, randtype = self.rtype)
            elif self.rtype == 'Coord':
                u = ot.sampling(n_samp = 1, dim = self.n, randtype = self.dist_dir)

            # normalize
            u /= np.linalg.norm(u)
        fmin, xmin = self.CARS_step(u, self.r)
        self.x = xmin
        self.ximg = self.Atk.xmap(self.x)
        self.fval = fmin

        self.t += 1
        self.stopiter()
        # decrease p as #iter increases
        if self.t in [2,10,40,250,500,800,1200,1600]:
            self.p /= 2.
        if self.status != None:
            return self.function_evals, self.x, self.status
        else:
            return self.function_evals, None, None   