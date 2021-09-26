%% Simple Demo for CARS

% set up a sample function and a problem (target accuracy, etc.)
param = param_setup();
fparam = struct;
fparam.f = @(x) myFunction(x, param);
fparam.fmin = 0; % min f value
fparam.name = 'a simple, random quadratic function';

% run CARS
NQ = 5; % 0: regular CARS, otherwise number of points for numerical quadrature
% supported: NQ = 0, 3, 4, 5, 6, 7, 10. Use odd NQ for efficiency
result = CARS(fparam, param, NQ);

figure();
plot(result.num_queries, log10(result.objval_seq));
title(['Number of queries vs log10(f(x)) for ', fparam.name]);
xlabel('Number of queries');
ylabel('log10(f(x))');




%% Parameter Setting

% sample function parameters
function param = param_setup()
param = struct;
param.n = 10; % problem dim
param.eps = 1e-6; % target accuracy
param.x0 = ones(param.n,1); % initial point
param.maxit = 1e4; % max iterations
param.verbose = 1; % verbose option
param.MAX_QUERIES = 3e4; % max number of function queries

% used for setting up a simple test function
A = randn(param.n); A = A*A' + 0.1*eye(param.n);
param.A = A;
end

% a simple quadratic function
function y = myFunction(x, param)
y = dot(x,param.A*x);
end