%% setup f
if ~again
    lambda = 1;
    MORETEST = strcmp(ProbType, 'MOR') || strcmp(ProbType,'NMO');
    if MORETEST
        addpath('MoreGarbowHillstrom');
        if ftn == 1 % badscb
            fname = 'badscb';
            n = 2; m = 3; x0 = [1,1]'; fmin = 0; true_sol = [1e6,2e-6]';
            f_M = @(x) badscb(n,m,x,1);
        elseif ftn == 2 % badscp
            fname = 'badscp';
            n = 2; m = 2; x0 = [0,1]'; fmin = 0; true_sol = [1.098e-5, 9.106]';
            f_M = @(x) badscp(n,m,x,1);
        elseif ftn == 3 % band
            fname = 'band';
            n = 100; m = n; x0 = -ones(n,1); fmin = 0; %true_sol = [1.098e-5, 9.106]';
            f_M = @(x) band(n,m,x,1);
        elseif ftn == 4 % bard
            fname = 'bard';
            n = 3; m = 15; x0 = ones(n,1); fmin = 8.21487e-3; %true_sol = [1.098e-5, 9.106]';
            f_M = @(x) bard(n,m,x,1);
        elseif ftn == 5 % bd
            fname = 'bd';
            n = 4; m = 20; x0 = [25, 5, -5, -1]'; fmin = 85822.2; %true_sol = [1.098e-5, 9.106]';
            f_M = @(x) bd(n,m,x,1);
        elseif ftn == 6 % beale
            fname = 'beale';
            n = 2; m = 3; x0 = [1,1]'; fmin = 0; true_sol = [3, 0.5]';
            f_M = @(x) beale(n,m,x,1);
        elseif ftn == 7 % biggs
            fname = 'biggs';
            n = 6; m = 13; x0 = [1,2,1,1,1,1]'; fmin = 0; true_sol = [1,10,1,5,4,3]';
            f_M = @(x) biggs(n,m,x,1);
        elseif ftn == 8 % box
            fname = 'box';
            n = 3; m = 10; x0 = [0,10,20]'; fmin = 0; true_sol = [1,10,1]';
            % other sols = [10,1,-1), and (t, t, 0) for all t
            f_M = @(x) box(n,m,x,1);
        elseif ftn == 9 % bv
            fname = 'bv';
            n = 100; m = n;
            tmp = (1:n)/(n+1); x0 = zeros(n,1);
            for test_idx=1:n
                x0(test_idx) = tmp(test_idx)*(tmp(test_idx)-1);
            end
            fmin = 0;
            f_M = @(x) bv(n,m,x,1);
        elseif ftn == 10 % froth
            fname = 'froth';
            n = 2; m = 2; x0 = [0.5, -2]'; fmin = 0; true_sol = [5, 4]';
            f_M = @(x) froth(n,m,x,1);
        elseif ftn == 11 % gauss
            fname = 'gauss';
            n = 3; m = 15; x0 = [0.4, 1, 0]'; fmin = 1.12793e-8;
            f_M = @(x) gauss(n,m,x,1);
        elseif ftn == 12 % gulf
            fname = 'gulf';
            n = 3; m = 100; x0 = [5, 2.5, .15]'; fmin = 0;
            f_M = @(x) gulf(n,m,x,1);
        elseif ftn == 13 % helix
            fname = 'helix';
            n = 3; m = 3; x0 = [-1, 0, 0]'; fmin = 0;
            f_M = @(x) helix(n,m,x,1);
        elseif ftn == 14 % ie
            fname = 'ie';
            n = 100; m = n;
            tmp = (1:n)/(n+1); x0 = zeros(n,1);
            for test_idx=1:n
                x0(test_idx) = tmp(test_idx)*(tmp(test_idx)-1);
            end
            fmin = 0;
            f_M = @(x) ie(n,m,x,1);
        elseif ftn == 15 % jensam
            fname = 'jensam';
            n = 2; m = 10; x0 = [0.3, 0.4]'; fmin = 124.362;
            f_M = @(x) jensam(n,m,x,1);
        elseif ftn == 16 % kowosb
            fname = 'kowosb';
            n = 4; m = 11; x0 = [.25, .39, .415, .39]'; fmin = 3.07505e-4;
            f_M = @(x) kowosb(n,m,x,1);
        elseif ftn == 17 % lin
            fname = 'lin';
            n = 100; m = n; x0 = ones(n,1); fmin = m-n;
            f_M = @(x) lin(n,m,x,1);
        elseif ftn == 18 % lin0
            fname = 'lin0';
            n = 100; m = n; x0 = ones(n,1); fmin = (m*m+3*m-6)/2/(2*m-3);
            f_M = @(x) lin0(n,m,x,1);
        elseif ftn == 19 % lin1
            fname = 'lin1';
            n = 100; m = n; x0 = ones(n,1); fmin = (m*(m-1))/2/(2*m+1);
            f_M = @(x) lin1(n,m,x,1);
        elseif ftn == 20 % meyer
            fname = 'meyer';
            n = 3; m = 16; x0 = [.02, 4000, 250]'; fmin = 87.9458;
            f_M = @(x) meyer(n,m,x,1);
        elseif ftn == 21 % osb1
            fname = 'osb1';
            n = 5; m = 33; x0 = [.5, 1.5, -1, .01, .02]'; fmin = 5.46489e-5;
            f_M = @(x) osb1(n,m,x,1);
        elseif ftn == 22 % osb2
            fname = 'osb2';
            n = 11; m = 65; x0 = [1.3,0.4, 0.65,0.7,0.6,3,5,7,2,4.5,5.5]'; fmin = 4.01377e-2;
            f_M = @(x) osb2(n,m,x,1);
        elseif ftn == 23 % pen1
            fname = 'pen1';
            n = 10; m = n+1; x0 = (1:n)'; fmin = 7.08765e-5;
            f_M = @(x) pen1(n,m,x,1);
            
        elseif ftn == 24 % pen2
            fname = 'pen2';
            n = 10; m = 2*n; x0 = 0.5*ones(n,1); fmin = 2.93660e-4;
            f_M = @(x) pen2(n,m,x,1);
        elseif ftn == 25 % rosen
            fname = 'rosen';
            n = 2; m = 2; x0 = [1.1, 1.1]'; fmin = 0;
            f_M = @(x) rosen(n,m,x,1);
        elseif ftn == 26 % rosex
            fname = 'rosex';
            n = 100; m = n;
            x0 = zeros(n,1);
            for test_idx=1:n
                if rem(test_idx,2)==0
                    x0(test_idx) = 1;
                else
                    x0(test_idx) = -1.2;
                end
            end
            fmin = 0;
            f_M = @(x) rosex(n,m,x,1);
        elseif ftn == 27 % sing
            fname = 'sing';
            n = 4; m = 4; x0 = [3, -1, 0, 1]'; fmin = 0;
            f_M = @(x) sing(n,m,x,1);
        elseif ftn == 28 % singx
            fname = 'singx';
            n = 100; m = n;
            x0 = zeros(n,1);
            for test_idx=1:n
                if rem(test_idx,4)==1
                    x0(test_idx) = 3;
                elseif rem(test_idx,4)==2
                    x0(test_idx) = -1;
                elseif rem(test_idx,4)==3
                    x0(test_idx) = 0;
                elseif rem(test_idx,4)==4
                    x0(test_idx) = 1;
                end
            end
            fmin = 0;
            f_M = @(x) singx(n,m,x,1);
        elseif ftn == 29 % trid
            fname = 'trid';
            n = 100; m = n; x0 = -ones(n,1); fmin = 0;
            f_M = @(x) trid(n,m,x,1);
        elseif ftn == 30 % trig
            fname = 'trig';
            n = 100; m = n; x0 = 1/n*ones(n,1); fmin = 0;
            f_M = @(x) trig(n,m,x,1);
        elseif ftn == 31 % vardim
            fname = 'vardim';
            n = 100; m = n+2; x0 = 1 - (1:n)'/n; fmin = 0;
            f_M = @(x) vardim(n,m,x,1);
        elseif ftn == 32 % watson
            fname = 'watson';
            n = 20; m = 31; x0 = zeros(n,1); fmin = 2.48631e-20;
            f_M = @(x) watson(n,m,x,1);
        elseif ftn == 33 % wood
            fname = 'wood';
            n = 4; m = 6; x0 = -[3,1,3,1]'; fmin = 0;
            f_M = @(x) wood(n,m,x,1);
        elseif ftn == 34 % Brown almost-linear
            fname = 'bl';
            n = 100; m = n; x0 = 0.5*ones(n,1); fmin = 0;
            f_M = @(x) bl(n,m,x,1);
        end
        freq = 2*pi*5e1*ones(n,1);
        fx0 = sum(f_M(x0).^2);
    else
        n = 20;
        lambda = +0.01;%1e-2;
        rk = n;
        A = randn(n, rk); A = A*A'; A = A/trace(A)*n + lambda*eye(n);
        alpha =  0.1*ones(n,1);
        f_M = @(x) dot(alpha, x.^4) + 0.5*dot(x,A*x);
        fmin = 0; x0 = ones(n,1); x0 = x0/norm(x0)*2;
        freq = 2*pi*5e1*ones(n,1);
        fx0 = f_M(x0);
        fname = 'cvx quartic';
        noise_lvl = noise_lvl * (fx0-fmin);
    end
    
    
    if strcmp(ProbType,'NMO')
        noise_lvl = noise_lvl * (fx0-fmin);
        f = @(x) sum(f_M(x).^2) + noise(x, freq, noise_lvl*(fx0-fmin)) + noise_lvl*(fx0-fmin); % Noisy More
    elseif strcmp(ProbType, 'MOR')
        f = @(x) sum(f_M(x).^2); % Noiseless More
    else
        f = @(x) sum(f_M(x)) + noise(x, freq, noise_lvl*(fx0-fmin)) + noise_lvl*(fx0-fmin); % Noisy quartic
    end
    
    fparam = struct;
    fparam.f = f;
    fparam.fmin = fmin;
    fparam.name = fname;
end

%% setup algorithm parameters

if ~again
    param = struct;
end

param.n = n;
fmin = fparam.fmin;
% EPS = 1e-5; % 1e-1; 1e-3; 1e-5;
if ~again
    param.x0 = x0;
end

f0 = fx0; %f(param.x0);
param.eps = EPS*(f0-fmin); % STP paper
param.eps_bergou = EPS;
param.maxit = maxit;
param.MAX_QUERIES = budget; %2*param.maxit for STP, for instance
param.verbose = verbose;
param.randAlg = 'G'; % Sample directions from Gaussian distribution

% for SPSA/2SPSA
param.tol1 = Inf; % prevents too large steps
param.tol2 = 0; % min decrease
if verbose>1
    disp('Parameter Settings are done!');
end
function e = noise(x, freq, noise_lvl)
if noise_lvl>0
    n = length(x);
    e = noise_lvl*1/n*(n-sum(cos(freq.*x)));
else
    e = 0;
end
end
