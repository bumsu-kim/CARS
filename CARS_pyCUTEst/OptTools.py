import numpy as np
import numpy.matlib
from scipy.special import roots_hermite

def sampling(n_samp, dim, randtype = 'Gaussian', distparam = None):
    '''
    returns a matrix of (n_samp) vectors of dimension (dim)
    Each vector is sampled from a distribution determined by (randtype) and (distparam)
    @ Params
        n_samp ..... number of sample vectors
        dim ........ dimension of each vector
        randtype ... (str) type of the distribution
                     possible types:
                     (i) Gaussian
                     (ii) Uniform

        distparam .. (dict) distributrion parameters
                     (i) Gaussian: {'mean' = mu, 'std' = sigma}
                     (ii) Uniform: {'mean' = x0 (vector), 'rad' = r}
                            currently only supports a unif dist on a sphere
                            centered at x0, with radius r
                     (iii) Coord : Random coordinate base
                            draw uniformly from {e_1, ..., e_n}
    @ Return 
        samples .... (np.ndarray) matrix of sampled vectors
                     size = (n_samp) - by - (dim)
    '''
    if randtype == 'Gaussian' or randtype == 'Normal':
        if distparam == None:
            return np.random.normal(size = (n_samp, dim)) # Gaussian
        else:
            return np.random.normal(distparam['mean'], distparam['std'], size = (n_samp, dim)) # Gaussian
    elif randtype == 'Uniform':
        mat = np.random.normal(size = (n_samp, dim)) # Gaussian
        norms = np.linalg.norm(mat,axis=1) # to Normalize
        if distparam == None:
            return mat/norms[:,None]
        else:
            return mat/norms[:,None]*distparam['rad'] + np.matlib.repmat(distparam['mean'],n_samp,1)
    elif randtype == 'Coord':
        if distparam == None:
            idx = np.random.choice(range(dim), size=(n_samp,1))
            mat = np.zeros((n_samp, dim))
            for i in range(n_samp):
                mat[idx[i]] = 1.
            return mat
    elif randtype == 'Box': # square atk
        wsz = distparam['windowsz'] # size of the square
        coord = distparam['coord'] # lower left corner of the square
        Isz = distparam['ImSize']
        mat = np.zeros(dim)
        sgn = np.random.rand()
        if sgn>0.5:
            sgn = 1
        else:
            sgn = -1
        for imdim in range(2):
            if coord[imdim]+wsz >= Isz[imdim]:
                coord[imdim] = Isz[imdim]-wsz-1
        for i in range(wsz):
            for j in range(wsz):
                mat[coord2idx(coord[0]+i, coord[1]+j)] = sgn
        return mat
    elif randtype == 'Vert': # random vertical perturabtion
        Isz = distparam['ImSize']
        mat = np.zeros(dim)
        for i in range(Isz[0]):
            sgn = np.random.rand()
            if sgn>0.5:
                sgn = 1
            else:
                sgn = -1
            for j in range(Isz[1]):
                mat[coord2idx(i,j)] = sgn
        return mat


def coord2idx(x, y, ImSize=(28, 28)):
    if 0<= y and y <= ImSize[0]:
        if 0<= x and x <= ImSize[1]:
            return x*ImSize[0] + y
    print(f"ERR (coord2idx): ({x}, {y}) cannot be converted into an idx")
    return 0

def idx2coord(idx, ImSize=(28,28)):
    if 0<= idx and idx <= ImSize[0]*ImSize[1]:
        return [idx//ImSize[0], idx%ImSize[0]]
    print(f"ERR (idx2coord): {idx} cannot be converted into coordinates")
    return (0,0)


def oracles(f, xs, pool=None):
    m = xs.shape[0]
    if pool==None:
        return np.array([f(xs[i,:]) for i in range(m)])
    else: # overhead can be large, so be careful
        xlist = [xs[i,:] for i in range(m)]
        return np.array(pool.map(f, xlist))

def oracles_with_p(f, xs, pool=None):
    m = xs.shape[0]
    if pool==None:
        val = np.empty(m)
        probs = np.empty((m,10))
        for i in range(m):
            res = f(xs[i,:])
            val[i] = res['fval']
            probs[i,:] = res['pdist'][0]
        return val, probs 
    # else part not implemented
    # else: # overhead can be large, so be careful
    #     xlist = [xs[i,:] for i in range(m)]
    #     return np.array(pool.map(f, xlist))

def CentDiff(f, x, h, u, fval, eps, proj = None):
    xp = x + h*u # proj(x + h*u)
    xm = x - h*u # proj(x - h*u)
    fp = f(xp)
    fm = f(xm)
    d = (fp-fm)/2./h
    d2 = (fp - 2*fval + fm)/h**2
    d2 = d2 + np.sign(d2)*eps
    return d, d2

def NumQuad(f, x, h, u, fval, ATK, GH_pts = 5):
    gh = roots_hermite(GH_pts)
    gh_value = np.expand_dims(gh[0], axis=1)
    if GH_pts%2 == 0:
        xs = np.matlib.repmat(x, GH_pts, 1) + h*np.sqrt(2.0)*gh_value*u
        fs = oracles(f,xs)
    else: # can reuse fval = f(x)
        xs = np.matlib.repmat(x, GH_pts, 1) + h*np.sqrt(2.0)*gh_value*u
        xm = xs[:(GH_pts-1)//2,:]
        xp = xs[-(GH_pts-1)//2:,:]
        fs = np.empty(GH_pts)
        fs[:(GH_pts-1)//2] = oracles(f,xm)
        fs[(GH_pts-1)//2] = fval
        fs[-(GH_pts-1)//2:] = oracles(f,xp)
    gh_weight = gh[1]
    fsgh = fs * gh_weight
    gh_value = np.transpose(gh_value)
    grad_u = 1. / np.sqrt(np.pi) / h    * np.sum(fsgh * (np.sqrt(2.)*gh_value))
    hess_u = 1. / np.sqrt(np.pi) / h**2 * np.sum(fsgh * (2*gh_value**2-1)) 
    D3f_u = 1. / np.sqrt(np.pi) / h**3 * np.sum(fsgh * (np.sqrt(8.)*gh_value**3-3.*np.sqrt(2.)*gh_value))
    D4f_u = 1. / np.sqrt(np.pi) / h**4 * np.sum(fsgh * (4*gh_value**4-6*2*gh_value**2+3))

    return grad_u, hess_u, D3f_u, D4f_u

def argminF(*args):
    '''
    @param *args = tuple of dicts, each of which containing 'x' and 'fval' fields
    @return dict with smallest fval
    '''
    minelt = args[0]
    minfval = minelt['fval']
    minidx = 0
    for idx, elt in enumerate(args):
        if elt['fval'] < minfval:
            minfval = elt['fval']
            minelt = elt
            minidx = idx
    return minelt, minidx
