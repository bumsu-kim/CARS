function [Tau, Rho, r] = performance_profile(algnames, evals, budget, rM, xlims, eps, showplot)
[nOpts, nftn] = size(evals);
r = zeros(size(evals));
for p = 1:nftn
    min_t = min(evals(:,p));
    for s = 1:nOpts
        if evals(s,p) >= budget
            r(s,p) = rM;
        else
            r(s,p) = evals(s,p)/min_t;
        end
    end
end

Tau = cell(nOpts,1);
Rho = cell(nOpts,1);
if showplot
    figure();
    hold on;
    linewidth = 2;
    fontsize = 20;
    opts_to_display = 1:length(Rho);
%     opts_to_display = [1,2,3,4,5,6,7,8]; % all
    for s = opts_to_display
        rs = r(s,:);
        [Tau{s}, Rho{s}] = create_tau_rho(rs);
        if s==1 || s==2 % CARS, CARS-NQ
            line_spec = '-.*';
        elseif s ==3 || s ==4 % STP,SMTP
            line_spec = '-.+';
        elseif s==6 || s==7 % SPSA, 2SPSA
            line_spec = '--o';
        else % others
            line_spec = '-';
        end
        plot(log2(Tau{s}), Rho{s}, line_spec, 'LineWidth', linewidth);
    end
    legend(algnames{opts_to_display}, 'Location', 'best');
    xlim(xlims);
    xlabel('log_2(\tau)', 'FontSize', fontsize);
    ylabel('\rho', 'FontSize', fontsize+4);
    title(['Performance Profile, ', char(949), '=',num2str(eps,'%g')], 'FontSize', fontsize);
else
    for s = 1:nOpts
        rs = r(s,:);
        [Tau{s}, Rho{s}] = create_tau_rho(rs);
    end
end
end








