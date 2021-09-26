% plot performance profile
function plot_res(tau, rho, eps, xlims)
figure();
    hold on;
    linewidth = 2;
    nOpts = length(tau);
    for s = 1:nOpts
        plot(log2(tau{s}), rho{s}, 'LineWidth', linewidth);
    end
    xlim(xlims);
    title(['Performance Profile, ', char(949), '=',num2str(eps,'%g')]);
    xlabel('log_2(\tau)');
    ylabel('\rho');
    
    legend('CARS', 'CARS-NQ', 'STP-fs', 'STP-vs', 'SMTP', 'Nesterov', 'Location','best');
%     legend('CARS', 'CARS-adaMu', 'CARS-NQ', 'STP-fs', 'STP-vs', 'SMTP', 'Nesterov', 'Location','best');
%     legend('CARS', 'CARS-NQ',  'STP-vs', 'SMTP', 'Location', 'best');
%     title(['Performance Profile, \vareps = ', num2str(eps)]);
end