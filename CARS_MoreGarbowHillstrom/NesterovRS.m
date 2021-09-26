function Result = NesterovRS(fparam, param)
algname = 'Nesterov-Spokoiny';
%% INITIALIZATION
Result = struct;
n = param.n;
eps = 1e-6; maxit = 100;

x = zeros(n,1); % initial sol (x0)
randAlg = 'U'; % uniform distribution
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
if isfield(param, 'randAlg')
    randAlg = param.randAlg;
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
mu_init = 1e-4;
mu = mu_init;

mu_seq = zeros(maxit+1,1);
mu_seq(1) = mu;
alpha = 1/4/(n+4); % used the same parameter in STP paper (bergou et al. 2019)

for k=1:maxit
    num_queries(k+1) = num_queries(k);
    mu =  1/sqrt(k+1);
    fx = objval_seq(k); % use saved value
    u = PickRandDir(1, n, randAlg)'; 
    u = u/norm(u); % normalize u
    fp = f(x+mu*u);
    num_queries(k+1) = num_queries(k+1) + 1;
    d = (fp - fx)/mu; % directional derivative
    
    delta = -alpha*d*u; % move to the next iterate
    
    fxnew = f(x+delta);
    num_queries(k+1) = num_queries(k+1) + 1; %single query here
    
    fs = [fx, fp, fxnew];
    [fxnew, midx] = min(fs);
    if midx == 1
        % no update
        delta = 0;
    else
        if midx == 2
            delta = mu*u;
        elseif midx == 3
            % delta not changed
        end
    end
    
    x = x + delta; % update the solution
    
    if isnan(fxnew) % should never happen..
        disp('error -- f(x) from NS is nan');
        x = param.x0;
        continue;
    end
    
    objval_seq(k+1) = fxnew;
    
    if isfield(fparam, 'fmin')
        if (fxnew < fmin + eps) 
            if verbose>1
                disp([algname, ' Converged in ', num2str(k),' steps. Exit the loop']);
                disp(['Function val = ' , num2str(fxnew)]);
            end
            converged = true;
            break;
        end
    end
    if (num_queries(k+1)>param.MAX_QUERIES)
        break;
    end
    mu_seq(k+1) = mu;
end

if (k>=maxit) || (num_queries(k+1)>param.MAX_QUERIES)
    if verbose>1
        disp([algname, ' did not converge in ', num2str(maxit) , ' steps.']);
    end
    converged = false;
end

num_iter = k;
objval_seq = objval_seq(1:num_iter+1);
num_queries = num_queries(1:num_iter+1);
mu_seq = mu_seq(1:num_iter+1);

% put into a struct for output
Result.objval_seq = objval_seq;
Result.num_iter = num_iter;
Result.num_queries = num_queries;
Result.converged = converged;
Result.mu_seq = mu_seq;
Result.sol = x;
end
