function Psi = find_intersection_line_plane(plane,line,line_point)

plane_normal = plane(1:3)';
ndotu = dot(plane_normal,line);

epsilon = 1e-6;
if abs(ndotu) < epsilon
    disp("no intersection between line and plane")
end

plane_point = [0, 0, -plane(3)/plane(2)];

% u = line
% n = plane_normal

w = line_point - plane_point;

si = dot(-plane_normal,w) / dot(plane_normal,line);

Psi = w + si*line;

end