%---------------------------
% Run the same experiment again?
again = false;
%---------------------------

ProbType = 'MOR'; % More-Garbow-Hillstrom 34 problems
% ProbType = 'NMO'; % Noisy More-Garbow-Hillstrom 34 problems
% ProbType = 'QUA'; % quartic, noisy
Param_Settings;

tic;
Results = cell(0,1);

etime2=zeros(0,1);

i = 0;
algname = cell(0,1);

param.eps_dep_mu = false;
param.adaptive_Lhat = false;
param.coord_change = false;

%% CARS
i = i+1;
algname{i} = 'CARS';
if verbose>1
    disp(['Starting ', algname{i}, ' ...']);
end
tic;
Results{i} = CARS(fparam, param, 0);
etime2(i) = toc;
Results{i}.name = algname{i};

%% CARS - NQ
i = i+1;
algname{i} = 'CARS-NQ';
if verbose>1
    disp(['Starting ', algname{i}, ' ...']);
end
tic;
Num_Quad_Pts = 5;
Results{i} = CARS(fparam, param, Num_Quad_Pts);
etime2(i) = toc;
Results{i}.name = algname{i};

%% STP - variable step size
i = i+1;
param.fixed_step = false;
algname{i} = 'STP-vs';
if verbose>1
    disp(['Starting ', algname{i}, ' ...']);
end
Results{i} = SMTP(fparam, param, algname{i}, false);
Results{i}.name = algname{i};

%% SMTP
i = i+1;
algname{i} = 'SMTP';
if verbose>1
    disp(['Starting ', algname{i}, ' ...']);
end
Results{i} = SMTP(fparam, param, algname{i}, true);
Results{i}.name = algname{i};


%% Nesterov-Spokoiny for comparison (RGF)

i = i+1;
algname{i} = 'Nesterov';
if verbose>1
    disp(['Starting ', algname{i}, ' ...']);
end
tic;
Results{i} = NesterovRS(fparam,param);
etime2(i) = toc;
Results{i}.name = algname{i};

%% SPSA method
i = i+1;
algname{i} = 'SPSA';
if verbose>1
    disp(['Starting ', algname{i}, ' ...']);
end
tic;
Results{i} = RealSPSA(fparam,param);
etime2(i) = toc;
Results{i}.name = algname{i};

%% 2SPSA
i = i+1;
algname{i} = '2-SPSA';
if verbose>1
    disp(['Starting ', algname{i}, ' ...']);
end
tic;
Results{i} = Real2SPSA(fparam,param);
etime2(i) = toc;
Results{i}.name = algname{i};

%% AdaDGS
i = i+1;
algname{i} = 'AdaDGS';
if verbose>1
    disp(['Starting ', algname{i}, ' ...']);
end
tic;
Num_Quad_Pts = 5;
Results{i} = AdaDGS(fparam, param, Num_Quad_Pts);
etime2(i) = toc;
Results{i}.name = algname{i};
