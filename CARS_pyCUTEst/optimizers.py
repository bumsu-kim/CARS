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
    def __init__(self, param, y0 = None, f = None):
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
        self.ishift = param['ishift'] # how much we shift the initial point from y0

        if y0 != None:
            self.sety0(y0)
        if f != None:
            self.setf(f)
        
        self.stationarity_threshold = param['threshold_first_order_opt']

        self.Atk = param['atk']
        self.function_budget = param['budget']
        self.function_target = param['target_fval']
        self.r = param['r'] # initial sampling radius

        self.fmin = float('inf')
        self.xmin = None

        self.constrained = param['constrained']
        self.nq = param['nq'] # numerical quadrature points
        
        # record where the min f value came from. (counts [x0, x +- ru, x_{CARS}, x_{boundary}])
        self.CARScounter = numpy.zeros(4, dtype=int)
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

    '''
    function evaluation
    '''
    def eval(self, f, x, record_min = True):
        ''' function evaluation at x
        '''
        self.curr_x = x
        res = f(self.curr_x)

        # record the minimum f
        if record_min:
            self.function_evals += 1 # count func evals
            if  res < self.fmin: # new min found
                self.xmin = x
                self.fmin = res

        return res

    
    

    '''
    Stopping Condition Check
    'B' -- Max budget reached
    'T' -- Target value reached
    'S' -- Stationary (reached First Order Optimality)
    '''
    def stopiter(self):
        # max budget reached
        if self.reachedFunctionBudget(self.function_budget, self.function_evals):
            self.status = 'B'

        # target reached
        if self.function_target != None:
            if self.reachedFunctionTarget(self.function_target, self.fval):
                self.status = 'T'

        if self.reachedFirstOrderOptimality(np.linalg.norm(self.grad(self.x)), self.stationarity_threshold):
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
        

        if self.nq == 0: # regular CARS
            d, d2 = ot.CentDiff(self.f, self.x, r, u, self.fval)
            Lhat = 1/2 + np.sqrt(1/4 + self.M*np.abs(d/d2**2/2))
            alpha = 1/Lhat
        else: # CARS-NQ
            d, d2, d3, d4 = ot.NumQuad(self.f, self.x, r, u, self.fval, self.Atk, self.nq)
            d3max = np.abs(d3) + np.abs(d/d2*d4) # estimate of sup|f'''| near x
            # 1/Lhat estimated from higher order derivatives
            alpha = self.alpha/ ( 0.5 + np.sqrt( 0.25 + np.abs(d*d3max/d2**2/3))) # see proposition 5.3

        if d2 > 1e-8: # when convex
            xnew, tnew = self.restrict_to_box(self.x, - alpha * d/d2 * u)
            tnew = tnew * (- alpha * d/d2 )
            self.cvx_counter += 1 # count this case
            fxnew = self.f(xnew)
        
        xnew1, tnew1 = self.restrict_to_box(self.x, - np.sign(d)*u, False) # move max amount along the direction of -d*u
        tnew1 = tnew1 * (- np.sign(d))
        fxnew1 = self.f(xnew1)
        tnew_bd = tnew1
        fxnew_bd = fxnew1
            
        if d2 <= 1e-8  or  fxnew_bd < fxnew:
            fxnew = fxnew_bd
            tnew = tnew_bd
            bd_chosen = True
        else:
            bd_chosen = False
            
        # self.check_dist(xnew)
        # check min
        if self.fmin == self.fval:
            minidx = 0 # not moved (xmin = x0)
        elif self.fmin == fxnew:
            if bd_chosen: # min found at the boundary
                minidx = 3
            else: # min is x_{CARS}
                minidx = 2
        else: # min is either x +- ru 
            minidx = 1
        self.CARScounter[minidx] += 1
       
        return self.fmin , self.xmin
    

    # def truncate_search_dir(self, u):
    #     v = np.copy(u)
    #     y = self.x + 1e-2*u # also works for u = [rows of directions]
    #     idx1 = np.logical_or(y>self.ub, y>self.Atk.data+self.Atk.eps)
    #     idx2 = np.logical_or(y<self.lb, y<self.Atk.data-self.Atk.eps)
    #     idx = np.logical_or(idx1, idx2)
    #     v[idx] = 0.
    #     if np.linalg.norm(v) == 0:
    #         return v
    #     v = v/np.linalg.norm(v)
    #     return v
        

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
        fmin, xmin = self.CARS_step(u, r)
        self.x = xmin
        self.fval = fmin

        self.t += 1
        self.stopiter()
        
        if self.status != None:
            return self.x, self.function_evals, True # 3rd val = termination or not
        else:
            return None, self.function_evals, False # 3rd val = termination or not
        #else:
        #    return self.function_evals, None, None   


#######################################################################################
## NEED UPDATES 

class SMTP(OptForAttack):
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