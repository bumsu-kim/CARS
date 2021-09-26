% Rosenbrock function 
% ------------------- 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [fvec,J]=rosen(n,m,x,option)
% Problem no. 1
% Dimensions -> n=2, m=2              
% Standard starting point -> x=(1,1)  
% Minima -> f=0 at (1,1)              
%                                     
% Revised on 10/22/94 by Madhu Lamba  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fvec,J] = rosen(n,m,x,opt)  
if (opt==1 | opt==3)
        fvec = [  10*(x(2)-x(1)^2)
                  1-x(1)  ] ; 
          else fvec='?';
end;        
if (opt==2 | opt==3)
        J    = [  -20*x(1)  10
                  -1        0  ] ; 
          else J='?';
end;
%
