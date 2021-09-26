function [fvec,J] = bl(n,m,x,opt)

% ***************************************
% Function [fvec,J]= bl(n,m,x,opt)
% Brown Almost-Linear Function
% Dimensions:    n=variable    m=n
% Function definitions:
%       fi(x)=xi + sum(x) - (n+1), 1 <= i < n
%       fn(x)= prod(x) - 1
% Starting point: (0.5, 0.5, ..., 0.5)
% Minima of f=0 at (t, t, ..., t, t^(1-n)), where
%                   nt^n - (n+1)t^(n-1) + 1 = 0.
%                   e.g. t = 1
%
% Coded in Matlab  June 23, 2021   Bumsu Kim
% **************************************8


if((opt ==1 )||(opt ==3))
    fvec = zeros(m,1);
    addthis = sum(x) - (n+1);
    for i=1:(m-1)
        fvec(i) = x(i) + addthis;
    end
    fvec(m) = prod(x) -1 ;
end
% 
% if((opt==2)|(opt==3))
% 
% J    =  [  1                           10       0                            0
%            0                            0       sqrt(5)               -sqrt(5)
%            0              2*(x(2)-2*x(3))      -4*(x(2)-2*x(3))              0 
%            2*sqrt(10)*(x(1)-x(4))       0       0      -2*sqrt(10)*(x(1)-x(4))];
%    else J='?';
% 
% end;
if((opt<1)|(opt>3))
        disp('Error: Option value for SING.M is either <1 or >3');
end;
