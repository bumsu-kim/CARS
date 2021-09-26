function Result = SMTP(fparam, param, algname, momentum)
% Search direction found in the orthogonal complement of the previous
% direction(s)

%% INITIALIZATION
Result = struct;
n = param.n;

x = zeros(n,1); % initial sol (x0)
randAlg = 'U'; % uniform
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

r_seq = zeros(maxit+1,1); % records the sampling radius (gamma)

beta = 0.5;
v = zeros(n,1);
for k=1:maxit
    
    num_queries(k+1) = num_queries(k);
    
    fx = objval_seq(k); % use saved value
    u = PickRandDir(1, n, randAlg)'; % first choose std normal
    u = u/norm(u);
    if param.fixed_step % fixed-step performs poorly
        gamma = 0.1*param.eps_bergou;
    else % use this
        gamma = 1/sqrt(k+1);
    end
    
    if momentum % SMTP
        vp = beta*v + u;
        vm = beta*v - u;
        xp = x - gamma*vp;
        xm = x - gamma*vm;
        zp = xp - gamma*beta/(1-beta)*vp;
        zm = xm - gamma*beta/(1-beta)*vm;
        
        fp = f(zp);
        fm = f(zm);
        
        [fx, idx] = min([fx, fp, fm]);
        if idx==1
            % no updates
        elseif idx==2
            x = xp;
            v = vp;
        elseif idx==3
            x = xm;
            v = vm;
        end
        
    else % STP
        fp = f(x+gamma*u);
        fm = f(x-gamma*u);
        [fx, idx] = min([fx, fp, fm]);
        if idx==1
            delta = 0;
        elseif idx==2
            delta = gamma*u;
        elseif idx==3
            delta = -gamma*u;
            
        end
        x = x+delta;
    end
    r_seq(k+1) = gamma;
    fxnew = fx;
    objval_seq(k+1) = fx;
    num_queries(k+1) = num_queries(k+1) + 2;
    
    if isfield(fparam, 'fmin')
        if (fx < fmin + eps)
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
end
if (k==maxit) || (max(num_queries)>param.MAX_QUERIES)
    if verbose>1
        disp([algname, ' did not converge in ', num2str(maxit) , ' steps.']);
    end
    converged = false;
end
num_iter = k;
objval_seq = objval_seq(1:num_iter+1);
num_queries = num_queries(1:num_iter+1);
r_seq = r_seq(1:num_iter+1);

% put into a struct for output
Result.objval_seq = objval_seq;
Result.num_iter = num_iter;
Result.num_queries = num_queries;
Result.converged = converged;
Result.mu_seq = r_seq;
Result.sol = x;
end





