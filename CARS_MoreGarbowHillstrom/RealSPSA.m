function Result = RealSPSA(fparam, param)
% The SPSA algorithm. As described in:
%   "Multivariate Stochastic Approximation Using a Simultaneous
%   Perturbation Gradient Approximation"
% by James C. Spall et al

%% INITIALIZATION
Result = struct;
n = param.n;
eps = 1e-6; maxit = 3000;
x = zeros(n,1);
verbose = false;

% Used the parameters suggested in the SPSA paper
alpha = 0.602;
gamma = 0.101;
A = 100;
a = 0.16;
c = 1e-4; % small if no noise

if isfield(param, 'eps')
    eps = param.eps;
end
if isfield(param, 'x0')
    x = param.x0;
end
if isfield(param, 'A')
    A = param.A;
end
if isfield(param, 'a')
    a = param.a;
end
if isfield(param, 'alpha')
    alpha = param.alpha;
end
if isfield(param, 'c')
    c = param.c;
end
if isfield(param, 'gamma')
    gamma = param.gamma;
end
if isfield(param, 'maxit')
    maxit = param.maxit;
end
if isfield(param, 'verbose')
    verbose = param.verbose;
end
if isfield(param, 'tol1')
    tol1 = param.tol1;
end
if isfield(param, 'tol2')
    tol2 = param.tol2;
end

if isfield(fparam, 'fmin')
    fmin = fparam.fmin;
end
f = fparam.f;
objval_seq = zeros(maxit+1,1);
sol_seq = cell(maxit+1,1);
fx = f(x);
objval_seq(1) = fx; %initialization
sol_seq{1} = x;
num_queries = zeros(maxit+1,1);
num_queries(1) = 1;
%% ITERATION
converged = false;
for k=1:maxit
    
    num_queries(k+1) = num_queries(k);
    
    Delta = PickRandDir(n,1,'R'); % no Unif/Gaussian allowed.
    % each elt of Delta = +-1 w.p. 1/2
    
    ak = a/(A+k+1)^alpha; % step size
    ck = c/(k+1)^gamma; % sampling radius
    fxp = f(x+ck*Delta); fxm = f(x-ck*Delta);
    gk = (fxp - fxm)./(2*ck*Delta); num_queries(k+1) = num_queries(k+1)+2;
    
    % update x
    xnew = x - ak*gk;
    fxnew = f(xnew);
    if fxnew > fx - tol2
        % do nothing;
    else % no blocking --> update
        x = xnew;
        fx = fxnew;
    end
    
    objval_seq(k+1) = fx;
    num_queries(k+1) = num_queries(k+1)+1;
    sol_seq{k+1} = x;
    
    if (fx < fmin + eps)% || norm(delta)<eps % or fx-fxnew < eps ?
        if verbose>1
            disp(['Real SPSA Converged in ', num2str(k),' steps. Exit the loop']);
            disp(['Function val = ' , num2str(objval_seq(k+1))]);
        end
        converged = true;
        break;
    end
    
    if (num_queries(k+1)>param.MAX_QUERIES)
        break;
    end
end

if (k>=maxit) || (num_queries(k+1)>param.MAX_QUERIES)
    if verbose>1
        disp(['Real SPSA did not converge in ', num2str(maxit) , ' steps.']);
    end
    converged = false;
end
num_iter = k;
objval_seq = objval_seq(1:num_iter+1);
num_queries = num_queries(1:num_iter+1);

% put into a struct for output
Result.objval_seq = objval_seq;
Result.num_iter = num_iter;
Result.num_queries = num_queries;
Result.converged = converged;
Result.sol = x;

end

