function plane = compute_plane(pts)
%   pts: 3x3, 3 3D points of form 3x1
% plane: 4x1, plane such that Ax+By+Cy+D=0

A = pts(:,1);
B = pts(:,2);
C = pts(:,3);

AB = B-A;
AC = C-A;

N = cross(AB,AC);

plane = [N(1); N(2); N(3); sum(-C.*N)];

end

