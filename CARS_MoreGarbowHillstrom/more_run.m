function [fevals, Results, finfo] = more_run(ftn, rep, EPS, noise_lvl, budget, verbose)
% if ~verbose
%     S = sprintf('opt = %d\t', opt);
%     fprintf(S);
% end
% nOpts = 3;
fevals = zeros(20, rep);
finfo = struct();
maxit = budget; % will never reach...
for j=1:rep
    moretest_script; % this sets the algorithms
    nOpts = length(Results);
    if verbose > 1
        fprintf('.')
    end
    for i = 1:nOpts
        fevals(i,j)= Results{i}.num_queries(end);
    end
end
fevals = fevals(1:nOpts, :);
if verbose > 1
    disp(['   ftn = ', num2str(ftn), ' done.'])
end
meanevals = mean(fevals,2);
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