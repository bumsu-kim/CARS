function Result = Real2SPSA(fparam, param)

%%
% Spall 2000
% 

%% INITIALIZATION
Result = struct;
n = param.n;
eps = 1e-6; maxit = 3000;
x = zeros(n,1);
verbose = false;

% Used the parameters suggested in the 2SPSA paper
alpha = 0.602;
gamma = 0.101;
A = 100;
a = 0.16;
c = 1e-4; % sampling radius; small if no noise
c2 = c*2; % sampling radius for 2nd order


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

Hk_bar = eye(n);
k_threshold = 1000;
warning('off','MATLAB:nearlySingularMatrix');
for k=1:maxit
    
    num_queries(k+1) = num_queries(k);
    
    Delta = PickRandDir(n,1,'R'); % no Unif/Gaussian allowed.
    Delta2 = PickRandDir(n,1,'R'); % no Unif/Gaussian allowed.
    % each elt of Delta = +-1 w.p. 1/2
    
    ak = a/(A+k+1)^alpha;
    ck = c/(k+1)^gamma;
    c2k = c2/(k+1)^gamma;
    fxp = f(x+ck*Delta); fxm = f(x-ck*Delta);
    gk = (fxp - fxm)./(2*ck*Delta); num_queries(k+1) = num_queries(k+1)+2;
    
    fxpp = f(x + ck*Delta + c2k*Delta2);
    gkp = (fxpp-fxp)./ (c2k * Delta2);
    fxmp = f(x - ck*Delta + c2k*Delta2);
    gkm = (fxmp - fxm) ./ (c2k*Delta2);
    num_queries(k+1) = num_queries(k+1)+2;
    dGk = gkp-gkm; % col vec
    Hk_hat = 1/2*(dGk'./(2*ck*Delta) + dGk./(2*ck*Delta'));
    
    Hk_bar = k/(k+1)*Hk_bar + Hk_hat/(k+1);
    if k<k_threshold
        Hk_dbar = F(Hk_bar); % make psd
    else
        Hk_dbar = Hk_bar;
    end
    if Hk_dbar == 0 % nan
        Hk_dbar = eye(n);
        Hk_bar = eye(n);
    end
    delta = -ak * (Hk_dbar\gk);
    xnew = x + delta;
    
    if norm(xnew-x) > tol1
        % do nothing
    else
       fxnew = f(xnew); num_queries(k+1) = num_queries(k+1)+1;
       if fxnew > fx - tol2
           % do nothing;
       else % no blocking --> update
           x = xnew;
           fx = fxnew;
       end
    end
    
    
    % update
    sol_seq{k+1} = x;
    objval_seq(k+1) = fx;
    
    if (fx < fmin + eps) % || norm(delta)<eps % or fx-fxnew < eps ?
        if verbose>1
            disp(['Real 2-SPSA Converged in ', num2str(k),' steps. Exit the loop']);
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
        disp(['Real 2-SPSA did not converge in ', num2str(maxit) , ' steps.']);
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


function Hbar = F(H)
n = size(H,1);
if any(isnan(H(:)))
    Hbar = 0;
    return;
end
% simple (for large n)
if n > 50
    E = eig(H); % n is still not too large in this test
    if min(diag(E))<0
        delta = -min(E(:))+sqrt(max(E(:)));
        Hbar = H + delta*eye(n);
    else
        Hbar = H;
    end
     return;
else
    delta = 0.01;
    % for small n
    M = H*H'+delta*eye(n);
    if any(isnan(M(:))) % error
        Hbar = 0;
        return;
    end
    Hbar = sqrtm(M);
end
if ~isreal(Hbar)
    Hbar = real(Hbar);
end
end

