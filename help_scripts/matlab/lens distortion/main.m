close all
clc
clear all
addpath(strcat(pwd,'/../scripts from colmap')); % add path to read_model() etc
addpath(strcat(pwd,'/../ransac to find plane')); % add path to compute and visualize plane etc


% Path to where points3D.txt, images.txt etc are located.
path = '/Users/ludvig/Documents/SSY226 Design project in MPSYS/test_workspace';

[cameras, images, points3D] = read_model(path);

img_keys = cell2mat(images.keys);

for key = img_keys(1)
    
    R = images(key).R;
    t = images(key).t;

    C = -R'*t;  % Camera center
    
    cam_id = images(key).camera_id;
    
    % Simple radial camera
    % params = f, cx, cy, k (from colmap code)
    pixelSize = 2.2/10^3;  % mm (2.2 μm, from cam spec)
    f  = cameras(cam_id).params(1); % in pixels
    %f  = 3 % in in mm [from lens spec]
    cx = cameras(cam_id).params(2); % principal point (in pixels)
    cy = cameras(cam_id).params(3); % principal point (in pixels)
    k  = cameras(cam_id).params(4); % (simple) radial distortion param

    % according to Fredrik:
    K = [f, 0, cx; 
         0, f, cy; 
         0, 0,  1];

    %Kinv = inv(K);

    P = K*[R,-R'*t];
    %Pc = [R,-R'*t];
    Pc = [R,C];

end

%% Test colmaps estimate of radial distortion

focalL = [f,f];
pPoint = [cx,cy];
imSize = [720,1280];
radDist = [k,k];  % k is usually -0.1507

int1 = cameraIntrinsics(focalL,pPoint,imSize,'RadialDistortion',radDist);

[J,newOri] = undistortImage(img1,int1);

figure(2)
subplot(2,1,1)
imshow(J)
subplot(2,1,2)
imshow(img1)

