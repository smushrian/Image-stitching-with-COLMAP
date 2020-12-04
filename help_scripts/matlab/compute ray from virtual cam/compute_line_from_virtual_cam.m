function [line_point,line_dir] = compute_line_from_virtual_cam(R,t,xy,cx,cy,f)
%  P: camera matrix on the form P = [R|t]
% xy: 2D pixel coordinates
% cx: principal point x
% cy: principal point y

x = xy(1);
y = xy(2);

C = -R'*t;

line_point = C';

vec_len = sqrt((x-cx)^2 + (y-cy)^2 + f^2);

line_dir = (C + R' * [x-cx; y-cy; f]./vec_len)' - C'; % unit vec

end