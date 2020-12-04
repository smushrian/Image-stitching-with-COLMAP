
clc
close all
clear all

% Parameters
syms k_1 k_2 real
syms r X_u Y_u
distX = subs(X_u * (1 + k_1 * r^2 + k_2 * r^4), r, sqrt(X_u^2 + Y_u^2));  % tangential dist: + 2 * p_1 * X_u * Y_u + p_2 * (r^2 + 2 * X_u^2)
distY = subs(Y_u * (1 + k_1 * r^2 + k_2 * r^4), r, sqrt(X_u^2 + Y_u^2));  % tangential dist: + 2 * p_2 * X_u * Y_u + p_1 * (r^2 + 2 * Y_u^2)

disp(expand(distX))
disp(expand(distY))

% Set Parameters
parameters = [k_1 k_2];
parameterValues = [-0.15 -0.15];
%plotLensDistortion(distX,distnY,parameters,parameterValues)

syms X_d Y_d positive
eq1 = X_d == distX;
eq2 = Y_d == distY;

eq1 = expand(subs(eq1, parameters, parameterValues));
eq2 = expand(subs(eq2, parameters, parameterValues));

disp('The equations to solve:')
disp(eq1)
disp(eq2)

res = solve([eq1, eq2], [X_u,Y_u], 'MaxDegree', 5,'Real',true,'ReturnConditions',true);

disp('Symbolic solve returned:')
disp([res.X_u res.Y_u])
disp('contstrained by the parameter(s):')
disp(res.parameters)
disp('such that')
disp(res.conditions)

% plot plane with X_d, Y_d and x somehow?



