close all
clc
clear all
addpath(strcat(pwd,'/../scripts from colmap')); % add path to read_model() etc

% Path to where points3D.txt, images.txt etc are located.
path = '/Users/ludvig/Documents/SSY226 Design project in MPSYS/test_workspace';

% Points3D
[cameras, images, points3D] = read_model(path);

img1 = im2double(imread('cam1.png'));
img2 = im2double(imread('cam2.png'));
img3 = im2double(imread('cam3.png'));
img4 = im2double(imread('cam4.png'));

% rgb2gray()

%% Plot 3D points
figure(1)
plot_model(cameras, images, points3D)
hold on;
xlabel('x');ylabel('y');zlabel('z');
axis([-4 10 -2 6 -5 10])

%% Extract 3D points and colors

N = length(points3D);
XYZ3D = zeros(3,N); % all 3D points now in a 3xN matrix rather than Map
Color3Dpoint = zeros(N,3); % color of all 3D points [Matlab color]

keys = cell2mat(points3D.keys);
for i = 1:N
    key = keys(i);
    XYZ3D(:,i) = points3D(key).xyz; 
    Color3Dpoint(i,:) = points3D(key).rgb' ./255; % RGB to Matlab color
end

%% Use Ransac to try to find floor plane
    
threshold = 0.2; % threshold distance from point to plane to count as an inlier
[plane, nbr_inliers] = ransac_find_plane(XYZ3D,threshold);

%% Vizualize found plane

A = plane(1);
B = plane(2);
C = plane(3);
D = plane(4);

[x, y] = meshgrid(-4:0.1:10); % Generate x and y data
z = -1/C*(A*x + B*y + D); % Solve for z data
surf(x,y,z) %Plot the surface

%% Compute a line from a given camera center and pose

% λu = R(U − C) = PU (and appending a 1 on U)

figure(1)
img_keys = cell2mat(images.keys);


for key = img_keys(1)
    R = images(key).R;
    t = images(key).t;

    C1 = -R'*t;  % Camera centre
    plot3(C1(1),C1(2),C1(3),'o','MarkerSize',30)
    
    
%     R = [0.7810,-0.0492,-0.6226;
%         -0.0057, 0.9963,-0.0858;
%          0.6245, 0.0706, 0.7778]
% 
%     t = [3.5546;
%          1.3919;
%          1.1106];
     
    R = [ 0.7810,-0.0492,-0.6226;
         -0.0057, 0.9963,-0.0858;
          0.6245, 0.0706, 0.7778];

    t = [-1.5546;
         -1.3919;
         2.1106];

    C = -R'*t
    plot3(C(1),C(2),C(3),'o','LineWidth',3,'MarkerSize',30)
    
    mid = [0.8766, 0.7783, 0.4192]
    
    cam_id = images(key).camera_id;
    
    % Simple radial camera
    % params = f, cx, cy, k (from colmap code)
    pixelSize = 2.2/10^3;  % mm (2.2 μm, from cam spec)
    f  = cameras(cam_id).params(1); % in pixels
    %f  = 3 % in in mm [from lens spec]
    cx = cameras(cam_id).params(2); % principal point (in pixels)
    cy = cameras(cam_id).params(3); % principal point (in pixels)
    k  = cameras(cam_id).params(4); % (simple) radial distortion param

    for idx_x = [0, 2*cx]
        for idx_y = [0, 2*cy]

            [line_point,line_dir] = compute_line_from_virtual_cam(R,t,[idx_x,idx_y],cx,cy,f);
            
            point_2 = line_point + 100*line_dir;
            line = [line_point;point_2];
            plot3(line(:,1),line(:,2),line(:,3),'r','LineWidth',1)
            
        end
    end
end


%mid_point = C1 + (C3-C1)./2;

%plot3(mid_point(1),mid_point(2),mid_point(3),'o','MarkerSize',30)

