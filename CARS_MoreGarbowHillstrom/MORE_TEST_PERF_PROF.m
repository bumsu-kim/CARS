%% Before running:
% First set algorithms to run in "moretest_script.m"
% Hyperparameters are set in "Param_Settings.m"


%% Problem Setting
ftns = 1:34; % More-Garbow-Hillstrom test functions
rep = 1; % repetitions to average
eps = 1e-1; % target accurcay (1e-1, 1e-3, 1e-5 for More), (1e-7 for Quartic)
budget = 2e3; % budget (default = 2e5)
verbose = 1; % verbose option (0, 1, 2)
showplot = 1; % show performance profile plot
noise_lvl = 0; % eps*1e-1; % noise level (default = 0), (default = eps*1e-1 for noisy case)

str_to_add = 'description_on_experiment_here'; % put a description on the current experiment here

rM = 1e20; % Use any sufficiently large number (default = 1e20)
xlims = [0, 12]; % x-axis limits

%% Solve and analyze
[EVALS, Results] = more_test(ftns, rep, eps, noise_lvl, budget, verbose);

save(['EVALS_',num2str(eps),'_',date,'_',str_to_add]);
disp(' ');

% save algorithm names
algnames = cell(length(Results),1);
for i=1:length(Results)
    algnames{i} = Results{i}.name;
end

% plot performance profiles
[tau, rho, r] = performance_profile(algnames, EVALS, budget, rM, xlims, eps, showplot);
save(['perf_prof_',num2str(eps),'_',num2str(noise_lvl),'_',date],'tau','rho');


%% Use this for plotting a Single function/problem result (e.g. Noisy Quartic)

% disp('Evaluation done. Plotting');
% figure(); hold on;
% Opts_to_disp = 1:length(Results);
% for s= Opts_to_disp
%     linewidth = 2;
%     if s==1 || s==2 % CARS, CARS-NQ
%         line_spec = '-.';
%     elseif s ==3 || s ==4 % STP,SMTP
%         line_spec = '-.';
%     elseif s==6 || s==7
%         line_spec = '--';
%     else
%         line_spec = '-';
%     end
%     if (s==3) % only provides the total number of queries
%         plot(1:2:Results{s}.num_queries, log10(Results{s}.objval_seq), line_spec, 'LineWidth', linewidth);
%     else 
%         plot(Results{s}.num_queries, log10(Results{s}.objval_seq), line_spec, 'LineWidth', linewidth);
%     end
% end
% fontsize=16;
% legend(algnames{Opts_to_disp}, 'Location', 'best','Orientation','vertical');
% xlim([0,budget]);
% xlabel('Function Queries', 'FontSize', fontsize);
% ylabel('$log_{10}(f(x))$', 'FontSize', fontsize, 'Interpreter','latex');