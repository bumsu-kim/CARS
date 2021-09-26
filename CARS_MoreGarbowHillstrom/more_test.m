function [evals, Results] = more_test(ftns, rep, eps, noise_lvl, budget, verbose)
nftn = length(ftns);
evals = zeros(20, nftn);

for i=1:nftn % run for each function
    [tempevals, Results, finfo] = more_run(ftns(i), rep, eps, noise_lvl, budget, verbose);
    nOpts = length(Results);
    evals(1:nOpts, i) = tempevals;
    
    % Brief Information on "failed" problems/algorithms
    if verbose > 0
        print_ftn_info = 0;
        for j = 1:nOpts
            print_ftn_info = print_ftn_info + (1-Results{j}.converged);
        end
        
        if print_ftn_info > 0
            fprintf('\n***************\nfunction name = %s\n', finfo.name);
            fprintf('\toptimal ftn value = %10.5e\n', finfo.fmin);
            fprintf('\tinitial ftn value = %10.5e\n', finfo.f0);
            fprintf('\ttarget ftn value = %10.5e\n\teps = %8.1e\n', eps*(finfo.f0 - finfo.fmin) + finfo.fmin,eps);
        end
        for j=1:nOpts
            status = Results{j}.converged > 0;
            if ~status
                fprintf('name: %s\n', Results{j}.name);
                fprintf('\t final ftn val = %10.5e\n',Results{j}.objval_seq(end));
                fprintf('\t (f(x)-fmin) / (f0-fmin) = %10.5e\n',(Results{j}.objval_seq(end)-finfo.fmin)/(finfo.f0-finfo.fmin));
                dist = norm(Results{j}.sol - finfo.x0);
                fprintf('\t distance moved from x0 = %10.5e\n', dist);
            end
        end
    end
end
evals = evals(1:nOpts,:);