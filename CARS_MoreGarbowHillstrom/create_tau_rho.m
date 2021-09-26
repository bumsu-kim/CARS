%% Helper function

% For a given algorithm, this function creates
%   the tau/rho vectors from its performance ratio
% notation:
%   P = # problems (idx: p)
%   s = idx for different algorithms
%
%   performance ratio for algorithm s := t(p,s) / min(t(p,s') over s');
%
% input : rvec (performance ratio vector, 1xP)
% output: tau
%         rho
%
function [tau, rho] = create_tau_rho(rvec)
    rvec = sort(rvec);
    P = length(rvec);
    tau = zeros(1, 2*P);
    rho = zeros(1, 2*P);
    
    tau(1) = 1; 
    rho(1) = 0; % start from (1,0)
    
    idx = 2;
    for ridx = 1:P % for each r value (from the smallest)
        if rvec(ridx) > tau(idx-1)
            tau(idx) = rvec(ridx);
            rho(idx) = ridx-1;
            idx = idx+1;
        end
        tau(idx) = rvec(ridx);
        rho(idx) = ridx;
        idx = idx+1;
    end
    tau = tau(1:idx-1);
    rho = rho(1:idx-1)/P;
    
%     tau(1) = 1; tau(2) = rvec(1);
%     rho(1) = 0; rho(2) = 0;
%     tidx = 1; % index for tau/rho
%     for ridx = 2:P
%         if tau(2*tidx) < rvec(ridx) % if r increased
%             tidx = tidx+1; % increment tau/rho index
%             tau(2*tidx-1) = rvec(ridx);
%             tau(2*tidx) = rvec(ridx); % add new r value
%             rho(2*tidx-1) = rho(2*tidx-2);
%             rho(2*tidx) = rho(2*tidx-2)+1; % add one
%         else % if r stays the same
%             rho(2*tidx) = rho(2*tidx)+1;
%         end
%     end
%     tau = tau(1:2*tidx);
%     rho = rho(1:2*tidx)/P;
end

