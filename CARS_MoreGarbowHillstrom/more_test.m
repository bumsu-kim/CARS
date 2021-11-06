function [evals, Results] = more_test(ftns, rep, eps, noise_lvl, budget, verbose)
nftn = length(ftns);
% evals = zeros(20, nftn);
Results = cell(nftn,1);
nOpts = 8; % Only for Nov6 test
evals = zeros(nOpts, nftn);
fprintf('Progress:\n');
fprintf(['\n' repmat('.',1,nftn) '\n\n']);
parfor i=1:nftn % run for each function
    [tempevals, Results{i}, finfo] = more_run(ftns(i), rep, eps, noise_lvl, budget, verbose);
%     nOpts = length(Results{i});
    evals(:, i) = tempevals;
    % Brief Information on "failed" problems/algorithms
    if verbose > 0
        print_ftn_info = 0;
        for j = 1:nOpts
            print_ftn_info = print_ftn_info + (1-Results{i}{j}.converged);
        end
        
        if print_ftn_info > 0
            fprintf('\n***************\nfunction name = %s\n', finfo.name);
            fprintf('\toptimal ftn value = %10.5e\n', finfo.fmin);
            fprintf('\tinitial ftn value = %10.5e\n', finfo.f0);
            fprintf('\ttarget ftn value = %10.5e\n\teps = %8.1e\n', eps*(finfo.f0 - finfo.fmin) + finfo.fmin,eps);
        end
        for j=1:nOpts
            status = Results{i}{j}.converged > 0;
            if ~status
                fprintf('name: %s\n', Results{i}{j}.name);
                fprintf('\t final ftn val = %10.5e\n',Results{i}{j}.objval_seq(end));
                fprintf('\t (f(x)-fmin) / (f0-fmin) = %10.5e\n',(Results{i}{j}.objval_seq(end)-finfo.fmin)/(finfo.f0-finfo.fmin));
                dist = norm(Results{i}{j}.sol - finfo.x0);
                fprintf('\t distance moved from x0 = %10.5e\n', dist);
            end
        end
    elseif verbose==0
        fprintf('\b|\n');
    end % no output when verbose<0
end
evals = evals(1:nOpts,:);