function res = compute_inv_lens_dist(k1,k2)

% Parameters
syms k_1 k_2 real
syms r X_u Y_u
distX = subs(X_u * (1 + k_1 * r^2 + k_2 * r^4), r, sqrt(X_u^2 + Y_u^2));  % tangential dist: + 2 * p_1 * X_u * Y_u + p_2 * (r^2 + 2 * X_u^2)
distY = subs(Y_u * (1 + k_1 * r^2 + k_2 * r^4), r, sqrt(X_u^2 + Y_u^2));  % tangential dist: + 2 * p_2 * X_u * Y_u + p_1 * (r^2 + 2 * Y_u^2)

% Set parameter values
parameters = [k_1 k_2];
parameterValues = [k1 k2];
%plotLensDistortion(distX,distnY,parameters,parameterValues)

syms X_d Y_d positive
eq1 = X_d == distX;
eq2 = Y_d == distY;

eq1 = expand(subs(eq1, parameters, parameterValues));
eq2 = expand(subs(eq2, parameters, parameterValues));

res = solve([eq1, eq2], [X_u,Y_u], 'MaxDegree', 5,'Real',true,'ReturnConditions',true);

end

