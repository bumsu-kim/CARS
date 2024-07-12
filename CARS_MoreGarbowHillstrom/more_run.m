function [fevals, Results, finfo,objval_plotting_data, objval_seq_matrix] = more_run(ftn, rep, EPS, noise_lvl, budget, verbose)
% if ~verbose
%     S = sprintf('opt = %d\t', opt);
%     fprintf(S);
% end
% nOpts = 3;
fevals = zeros(20, rep);
objval_seq_matrix = zeros(20, rep, budget); % store trajectories for every algorithm and every rep.
finfo = struct();
maxit = budget; % will never reach...
min_number_iters = maxit*ones(8,1); %fewest iterations observed per algorithm over all reps.
for j=1:rep
    moretest_script; % this sets the algorithms
    nOpts = length(Results); % this is the number of algorithms?
    if verbose > 1
        fprintf('.')
    end
    for i = 1:nOpts
        fevals(i,j)= Results{i}.num_queries(end);
        num_iters = length(Results{i}.objval_seq);
        objval_seq_matrix(i,j,1:num_iters) = Results{i}.objval_seq;
        if num_iters < min_number_iters(i)
            min_number_iters(i) = num_iters;
        end
    end
end
fevals = fevals(1:nOpts, :);
if verbose > 1
    disp(['   ftn = ', num2str(ftn), ' done.'])
end
meanevals = mean(fevals,2);
% process objective value trajectories for plotting quartic
% important to use logarithmic scale

mean_objval_seq = mean(objval_seq_matrix, 2);
min_objval_seq = min(objval_seq_matrix,[],2);
max_objval_seq = max(objval_seq_matrix,[],2);
objval_plotting_data{1} = mean_objval_seq;
objval_plotting_data{2} = min_objval_seq;
objval_plotting_data{3} = max_objval_seq;
objval_plotting_data{4} = min_number_iters;

if verbose>1
    medians = median(fevals, 2);
    for i = 1:nOpts
        disp([algname{i}, ' : avg = ', num2str(meanevals(i)), ', med = ', num2str(medians(i))]);
    end
end
fevals = meanevals;
finfo.fmin = fmin;
finfo.f0 = f0;
finfo.name = fname;
finfo.x0 = param.x0;
end