function residual_lengths = residual_lengths_points_to_plane(pts,plane)
%   pts: 3xN 3D points
% plane: 4x1 [a;b;c;d] such that ax+by+xz+d=0

% residual_lengths: 1xN, the minimum distance from all points to the plane

N = size(pts,2);
normal_vec = plane(1:3) ./ sqrt(sum(plane(1:3).^2));
residual_lengths = zeros(N,1);

% Is it rly this hard to find a point on a plane
if plane(1) ~= 0
    P = [-plane(4)/plane(1),0,0]'; 
elseif plane(2) ~= 0
    P = [0,-plane(4)/plane(2),0]'; 
elseif plane(3) ~= 0
    P = [0,0,-plane(4)/plane(3)]'; 
else
    P = [0,0,0]';
end

for i = 1:N
    u = pts(:,i) - P; % difference vector from plane to point
    residual_lengths(i) = abs(dot(u,normal_vec)); % length of difference vector projected onto normal vector
end


end

