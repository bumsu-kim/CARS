function r = getR(evals, budget, rM)
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
end