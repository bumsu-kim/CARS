function [delta, fmin] = Directional_Newton(f, x, P, h, k, adaptive_step, fx, safeguard)
%   Input:
%   f - function handle
%   x - current point
%   P - set of directions (n by m), 
%       if m == n is assumed, it must be an orthogonal matrix (P*P'=I)
%   h - smoothing parameter
%   k - # of quadrature points
%   fx - f(x) (may not be given)
%   safeguard - Use safeguard method or not

if nargin==5 && mod(k,2)==1
    fx = f(x);
end

[n,m] = size(P);
xbound_p = inf; %5.0;
xbound_m = -inf; %-5.0;
if m==n
    delta = zeros(n,1);
    alpha = 0.5*ones(n,1);
    [xi, wi] = GH_quad(k);
    fi = zeros(k,m); Di = zeros(k, m);
    for i=1:m
        v = P(:,i); % i-th direction
        [df,d2f,d3f,d4f, fi_tmp, Di_tmp] = GH_Deriv(k,xi,wi,f,h,v,x,fx);
        d3f_max = abs(d3f) + abs(df/d2f*d4f);
        if adaptive_step
            Lhat = 0.5 + sqrt( 0.25 + abs(df*d3f_max/d2f^2/3));
            %         Lhat = 2;
            alpha(i) = 0.5/Lhat;
        end
        delta(i) = -alpha(i)*df/d2f;
        fi(:,i) = fi_tmp;
        Di(:,i) = Di_tmp;
    end
    delta = P*delta;
    x_new = x+delta;
    % bound check
    % bound check
    if sum( x_new > xbound_p) + sum(x_new < xbound_m) > 0 % out of bound
        % then perform projection
        x_new(x_new > xbound_p) = xbound_p;
        x_new(x_new < xbound_m) = xbound_m;
        delta = x_new - x;
        fnew = f(x_new);
    else
        fnew = f(x+delta);
    end
    [minfi, minidx] = min(fi(:));
    if minfi < fnew
        iidx = floor((minidx-1)/k)+1;
        delta = Di(minidx)*P(:,iidx);
        fmin = minfi;
    else
        fmin = fnew;
    end
elseif m==1
    alpha = 0.5;
    [xi, wi] = GH_quad(k);
    
    v = P; % i-th direction
    [df,d2f,d3f,d4f, fi, Di] = GH_Deriv(k,xi,wi,f,h,v,x,fx);
    d2f = abs(d2f) + 1e-12;
    d3f_max = abs(d3f) + abs(df/d2f*d4f);
    if adaptive_step
        Lhat = 0.5 + sqrt( 0.25 + abs(df*d3f_max/d2f^2/3));
        %         Lhat = 2;
        alpha = 1/Lhat;
    end
    delta = -alpha*df/d2f;
    
    delta = P*delta;
    fnew = f(x + delta);
    if isnan(fnew)
%         disp('?');
    end
    [minfi, midx] = min(fi);
    if safeguard && minfi < fnew
        delta = Di(midx)*v;
        fmin = minfi;
    else
        fmin = fnew;
    end
end





