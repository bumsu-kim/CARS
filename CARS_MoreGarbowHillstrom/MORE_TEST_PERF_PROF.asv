%% Before running:
% First set algorithms to run in "moretest_script.m"
% Hyperparameters are set in "Param_Settings.m"


%% Problem Setting
% ftns = 1:34; % More-Garbow-Hillstrom test functions
ftns = 1:20;
rep = 1; % repetitions to average
eps = 0; 1e-7; % target accurcay (1e-1, 1e-3, 1e-5 for More), (1e-7 for Quartic)
budget = 2e6; % budget (default = 2e5)
verbose = 0; % verbose option (0, 1, 2)
showplot = 1; % show performance profile plot
noise_lvl = 0; % eps*1e-1; % noise level (default = 0), (default = eps*1e-1 for noisy case)

str_to_add = 'Nov6'; % put a description on the current experiment here

rM = 1e20; % Use any sufficiently large number (default = 1e20)
xlims = [0, 12]; % x-axis limits

%% Solve and analyze
% solve
[EVALS, Results] = more_test(ftns, rep, eps, noise_lvl, budget, verbose);
%% Save
save(['EVALS_',num2str(eps),'_',date,'_',str_to_add],'-v7.3');
disp(' ');

%% save algorithm names
algnames = cell(length(Results{1}),1);
for i=1:length(Results{1})
    algnames{i} = Results{1}{i}.name;
end

% % plot performance profiles
% [tau, rho, r] = performance_profile(algnames, EVALS, budget, rM, xlims, eps, showplot);
% save(['perf_prof_',num2str(eps),'_',num2str(noise_lvl),'_',date],'tau','rho');


%% Use this for plotting a Single function/problem result (e.g. Noisy Quartic)

disp('Evaluation done. Plotting');
% figure(); hold on;
ftn_num = 4;
Opts_to_disp = 1:length(Results{ftn_num});
% Opts_to_disp = [1, 2, 3, 4, 5, 6, 7, 8];
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
% %     if (s==3) % only provides the total number of queries
% %         plot(1:2:Results{s}.num_queries, Results{s}.objval_seq, line_spec, 'LineWidth', linewidth);
% %     else 
%         plot(Results{ftn_num}{s}.num_queries, Results{ftn_num}{s}.objval_seq, line_spec, 'LineWidth', linewidth);
% %     end
% end
% fontsize=16;
% legend(algnames{Opts_to_disp}, 'Location', 'best','Orientation','vertical');
% xlim([0,budget]);
% set(gca,'YScale','log')
% xlabel('Function Queries', 'FontSize', fontsize);
% ylabel('$f(x)$', 'FontSize', fontsize, 'Interpreter','latex');
%%
VariancePlot;

