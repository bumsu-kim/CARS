%% Variance Plot

% first gather them
fvals = cell(8,1);
qvals = cell(8,1);
means = cell(8,1);
nftns = length(ftns);
fvallow = cell(8,1);
fvalhigh = cell(8,1);
% CI = tinv([.05 .95], nftns-1);
Opts_to_disp = [1, 2, 3, 4, 5, 6, 7, 8];
low_percentile = 0.05;
high_percentile = 0.95;
for s = Opts_to_disp
    qvals{s} = Results{1}{s}.num_queries'; % row vec
    fvals{s} = zeros(nftns, length(qvals{s}));
    for j = ftns
        fvals{s}(j,:) = Results{j}{s}.objval_seq';
    end
    means{s} = mean(fvals{s});
    fvallow{s} = quantile(fvals{s}, low_percentile);
    fvalhigh{s} = quantile(fvals{s}, high_percentile);
%     fvalmed{s} = quantile(fvals{s}, 0.5);
%     sems{s} = std(fvals{s})/sqrt(nftns);
%     fCI{s} = bsxfun(@times, sems{s}, CI(:));
end
figure(); hold on;
set(gca,'YScale','log')
% fillparam = 100;
colors = {[0 0.4470 0.7410], [0.8500 0.3250 0.0980], [0.9290 0.6940 0.1250],[0.4940 0.1840 0.5560],[0.4660 0.6740 0.1880],[0.3010 0.7450 0.9330],[0.6350 0.0780 0.1840], [0 0.4470 0.7410]};
for s= Opts_to_disp
    linewidth = 2;
    if s==1 || s==2 % CARS, CARS-NQ
        line_spec = '-.';
    elseif s ==3 || s ==4 % STP,SMTP
        line_spec = '-.';
    elseif s==6 || s==7
        line_spec = '--';
    else
        line_spec = '-';
    end
    fillparam = floor(length(qvals{s})/100);
    if s==8 % adadgs
        fillparam = 1;
    end
    qv = qvals{s}(1:fillparam:end);
    fl = fvallow{s}(1:fillparam:end);
    fh = fvalhigh{s}(1:fillparam:end);
    fillobj = fill([qv, fliplr(qv)], [ fl, fliplr(fh)], colors{s});
    set(fillobj, 'facealpha',.3);
    set(fillobj, 'EdgeColor','None');
%     set(fillobj, 'FaceColor',colors{s});
    plot(qvals{s}, means{s}, line_spec, 'LineWidth', linewidth,'Color',colors{s});
%     plot(qvals{s}, fvallow{s}, line_spec, 'LineWidth', linewidth);
%     plot(qvals{s}, fvalhigh{s}, line_spec, 'LineWidth', linewidth);
%     plot(qvals{s}, fvalmed{s}, line_spec, 'LineWidth', linewidth);
%     plot(qvals{s}, means{s}+fCI{s}, line_spec, 'LineWidth', linewidth);
end
fontsize=16;
legendalgnames = cell(2*length(Opts_to_disp),1);
for i=1:length(Opts_to_disp)
    legendalgnames{2*i} = algnames{Opts_to_disp(i)};
    legendalgnames{2*i-1} = '';
end
% legend(legendalgnames, 'Location', 'best','Orientation','vertical');
xlim([0,budget]);
xlabel('Function Queries', 'FontSize', fontsize);
ylabel('$f(x)$', 'FontSize', fontsize, 'Interpreter','latex');
title('Averaged');

