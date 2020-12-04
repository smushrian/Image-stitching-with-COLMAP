function [x_d,y_d] = distort_coord(x_u,y_u,k1,k2)
% xy_u_: 2D coordinate (undistorted)
% k1,k2: radial distortion parameter

% xy_d: 2D coordinate (distorted)

x_d = x_u*(1 + k1*sqrt(x_u^2 + y_u^2)^2 + k2*sqrt(x_u^2 + y_u^2)^4);
y_d = y_u*(1 + k1*sqrt(x_u^2 + y_u^2)^2 + k2*sqrt(x_u^2 + y_u^2)^4);

end

