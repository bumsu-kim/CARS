function Z = PickRandDir(m, n, randAlg)
%
% ===== INPUT =====
% m         : number of measurements
% n         : dimension of each measurement vector
% randAlg   : How to pick random direction (U by default)
%   - 'U'   : pick each coordinate from Uniform(-1,1)
%   - 'US'  : pick uniformly from the Unit Sphere
%   - 'G''N': pick each coord from Gaussian normal (0,1)
%   - 'R'   : Each coord = +-1
%   - 'UC'  : Unif(coordinate vectors) (discrete)
% ===== OUTPUT =====
% Z     : m-by-n matrix
%         each row is a random direction (z_i)
%         z_i's normalized to have L2 norm 1.
Z = zeros(m,n);

if nargin == 2 % SET DEFAULT
    randAlg = 'U';
end
if strcmp(randAlg, 'U')
    Z = -1 + 2*rand(m,n);
    % normalize
    for i=1:m
        Z(i,:) = Z(i,:)/norm(Z(i,:));
    end
elseif strcmp(randAlg, 'G') || strcmp(randAlg, 'N')
    Z = randn(m,n);
elseif strcmp(randAlg, 'US')
    Z = randn(m,n);
    for i=1:m
        Z(i,:) = Z(i,:)/norm(Z(i,:));
    end
elseif strcmp(randAlg, 'R')
    Z = (rand(m,n)<.5)*2 - 1; % Rademacher dist for SPSA
elseif strcmp(randAlg, 'UC')
    coordids = randsample(n,m,true); % choose k values from 1:n with replacement
    Z = zeros(m,n);
    for j=1:length(coordids)
        Z(j,coordids(j)) = 1;
    end
end

end