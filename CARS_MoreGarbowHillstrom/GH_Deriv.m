function [df, d2f, d3f, d4f, fi, Di] = GH_Deriv(k, xi, wi, f, h, v, x, fx)

fi = zeros(k,1);
Di = zeros(k,1);
if mod(k,2) == 1 % odd # quadrature points
    fi((k+1)/2) = fx;
    % Di(:,(k+1)/2) = zeros
    for j=1:((k-1)/2)
        Di((k+1)/2 - j) = h * sqrt(2)* xi((k+1)/2 - j);
        Di((k+1)/2 + j) = h * sqrt(2)* xi((k+1)/2 + j);
        fi((k+1)/2 - j) = f(x + Di((k+1)/2 - j) * v );
        fi((k+1)/2 + j) = f(x + Di((k+1)/2 + j) * v );
    end
else % even # quadrature points
    for j=1:k
        Di(j) = h * sqrt(2)* xi(j) ;
        fi(j) = f(x + Di(j) * v);
    end
end

nv = norm(v);
% u
df  = 1/sqrt(pi)/(h*nv)   * sum( fi.*wi.*(sqrt(2)* xi));
% u^2 -1
d2f = 1/sqrt(pi)/(h*nv)^2 * sum( fi.*wi.*(2*xi.^2-1));
% u^3 - 3u
d3f = 1/sqrt(pi)/(h*nv)^3 * sum( fi.*wi.*(sqrt(8)*xi.^3 -3*sqrt(2)*xi));
% u^4 - 6u^2 + 3
d4f = 1/sqrt(pi)/(h*nv)^4 * sum( fi.*wi.*(4*xi.^4 - 6*2*xi.^2 + 3));

end