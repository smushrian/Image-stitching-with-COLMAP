close all
clc
clear all
addpath(strcat(pwd,'/../scripts from colmap')); % add path to read_model() etc

% Path to where points3D.txt, images.txt etc are located.
path = '/Users/ludvig/Documents/SSY226 Design project in MPSYS/test_workspace';

% Points3D
[cameras, images, points3D] = read_model(path);

%img1 = im2double(imread('cam1.png'));
%img2 = rgb2gray(im2double(imread('cam2.png')));
%img3 = im2double(imread('cam3.png'));
%img4 = im2double(imread('cam4.png'));

% rgb2gray()

%% Plot 3D points
plot_model(cameras, images, points3D)
hold on;
xlabel('x');ylabel('y');zlabel('z');
axis([-4 10 -2 6 -5 10])

%% Extract 3D points and colors

N = length(points3D);
XYZ3D = zeros(3,N); % all 3D points now in a 3xN matrix rather than Map
C = zeros(N,3); % color of all 3D points [Matlab color]

keys = cell2mat(points3D.keys);
for i = 1:N
    key = keys(i);
    XYZ3D(:,i) = points3D(key).xyz; 
    C(i,:) = points3D(key).rgb' ./255; % RGB to Matlab color
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

%% Compute projection of image down on plane

% λu = R(U − C) = PU (and appending a 1 on U)

% Restrict stichted image to a certain resolution
% GrandmasDialUp, potato, meh, 4k
resolution = 'ok';

switch resolution
    case 'meh'
        stichedImage = zeros(1080,1920);
    case '4k'
        stichedImage = zeros(2160,3840);
    otherwise
        error('Unrecognized image resolution')
end

C = images(1).t;
R = images(1).R;

% Simple radial camera
% params = f, cx, cy, k (from colmap code)
% Likely focal length, radial dist x, radial dist y,

f  = cameras(2).params(1); % in pixels
%f  = 3 % in in mm [from lens spec]
cx = cameras(2).params(2); % principal point (in pixels)
cy = cameras(2).params(3); % principal point (in pixels)
k  = cameras(2).params(4); % 

K = inv(Kinv);

%P = K[R|-RC], K IS NOT IMPLEMENTED YET
Pc = [R,-R*C];

% Pc*U = u_undist
% use k to get dist





