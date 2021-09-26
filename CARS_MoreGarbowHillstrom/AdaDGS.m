function Result = AdaDGS(fparam, param, n_quad_pts)
% The AdaDGS algorithm, as described in:
%   "AdaDGS: An Adaptive Black-Box Optimization Method with a Nonlocal 
%   Directional Gaussian Smoothing Gradient"
% By Huang Tran and Guannan Zhang, 2021
%
% Search direction found in the orthogonal complement of the previous
% direction(s)

%% INITIALIZATION
Result = struct;
n = param.n;
eps = 1e-6; maxit = 100;

x = zeros(n,1); % initial sol (x0)
verbose = false;

if isfield(param, 'eps')
    eps = param.eps;
end
if isfield(param, 'x0')
    x = param.x0;
end
if isfield(param, 'maxit')
    maxit = param.maxit;
end
if isfield(param, 'verbose')
    verbose = param.verbose;
end

if isfield(fparam, 'fmin')
    fmin = fparam.fmin;
end
f = fparam.f;
objval_seq = zeros(maxit+1,1);
objval_seq(1) = f(x); %initialization
num_queries = zeros(maxit+1,1);
num_queries(1) = 1;


%% ITERATION

mu = 1e-2; % initial sampling radius
rel_eps = 1e-3; % relative update tolerance
mu_0 = mu;
res_step = 10;

% Gauss-Hermite quadrature
mu_seq = zeros(maxit+1,1);
mu_seq(1) = mu;
ndelta = zeros(maxit,1);
n_LS = max(100, floor((n_quad_pts-1)*n*0.05)); % AdaDGS paper: dim = 1000. Since n << 1000 use larger default value here
[xi, wi] = GH_quad(n_quad_pts);

reset_id = 0;
for k=1:maxit
    num_queries(k+1) = num_queries(k);
    
    fx = objval_seq(k); % use saved value
    
    g = zeros(n,1);
    for ii=1:n
        ei = zeros(n,1); ei(ii) = 1;
        [dfi,~,~,~,~,~] = GH_Deriv(n_quad_pts, xi, wi, f, mu, ei, x, fx);
        g(ii) = dfi;
    end
    num_queries(k+1) = num_queries(k+1) + (n_quad_pts-1)*n;
    
    [fxnew, delta] = Simple_LS(f, x, g, fx, n_LS);
    num_queries(k+1) = num_queries(k+1) + n_LS;
    
    if isnan(fxnew) % should never happen..
        disp('error -- f(x) from adaDGS is nan');
        x = param.x0;
        continue;
    end
    
    mu_seq(k+1) = mu;
    ndelta(k) = norm(delta);
    x = x+delta; % update the solution
    objval_seq(k+1) = fxnew;
    
    rel_upd = (objval_seq(k+1) - objval_seq(k)) / objval_seq(k); % as in AdaDGS paper. in fact it doesn't work if f < 0
    
    if ((ndelta(k) == 0) || rel_upd < rel_eps)...
      && ...
       ((k - reset_id >= res_step) || k==1)
        mu = mu_0 + 0.1*randn*mu_0;
        reset_id = k;
    else
        mu = 0.5 * (mu + ndelta(k));
    end
    
    
    if isfield(fparam, 'fmin')
        if (fxnew < fmin + eps)
            if verbose>1
                disp(['AdaDGS Converged in ', num2str(k),' steps. Exit the loop']);
                disp(['Function val = ' , num2str(fxnew)]);
            end
            converged = true;
            break;
        end
    end
    if (max(num_queries)>param.MAX_QUERIES)
        break;
    end
end

if (k==maxit) || (max(num_queries)>param.MAX_QUERIES)
    if verbose>1
        disp(['AdaDGS did not converge in ', num2str(maxit) , ' steps.']);
    end
    converged = false;
end

num_iter = k;
objval_seq = objval_seq(1:num_iter+1);
num_queries = num_queries(1:num_iter+1);
mu_seq = mu_seq(1:num_iter+1);
ndelta = ndelta(1:num_iter);

% put into a struct for output
Result.objval_seq = objval_seq;
Result.sol = x;
Result.num_iter = num_iter;
Result.num_queries = num_queries;
Result.converged = converged;
Result.mu_seq = mu_seq;
Result.n_LS = n_LS;
Result.ndelta = ndelta;
end


function [fxnew, delta] = Simple_LS(f, x, g, fx, N)
n = length(x);
beta = sqrt(n)*2 / (norm(g)+1e-10); % max learning rate (rel to max search length)
tau  = 0.9; % contraction factor
fmin = fx;
delta_min = 0;
for i=1:N
    delta = - beta * tau^i * g;
    fxnew = f(x + delta);
    if fxnew < fmin % save min 
        fmin = fxnew;
        delta_min = delta;
    end
end
fxnew = fmin;
delta = delta_min;
end





