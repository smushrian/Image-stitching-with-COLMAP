from help_scripts.python_scripts.COLMAP_functions import *
from help_scripts.python_scripts.estimate_plane import *
from help_scripts.python_scripts.color_virtual_image import *
import numpy as np
import matplotlib.pyplot as plt
import os

# Perform the reconstruction to get data
automatic_reconstructor()
image_undistorter()
stereo_fusion()

cameras, points3D, images = get_data_from_binary()
# print(images)
coordinates = []
for key in points3D:
    coordinates.append(points3D[key].xyz)
coordinates = np.asarray(coordinates)

#Estimate a floor plane
plane, min_outlier = ransac_find_plane(coordinates, 0.01)

# Get all camera matrices and images
camera_intrinsics = {}
all_camera_matrices = {}
imgs = {}
# image_dir = '../COLMAP_w_CUDA/images/'
image_dir = '../COLMAP_w_CUDA/dense/0/images/'

for key in images:
    print('cameraid, name',images[key].camera_id,images[key].name)
    imgs[images[key].camera_id] = np.asarray(plt.imread(image_dir + images[key].name))
    all_camera_matrices[images[key].camera_id] = camera_quat_to_P(images[key].qvec, images[key].tvec)
    camera_intrinsics[cameras[key].id] = cameras[key]
# POSSIBLE VIRT CAMERA CENTER:
Pv = create_virtual_camera(all_camera_matrices)
# # print(Pv)
w = 300
h = 300
f = 100
K_virt = np.asarray([[f, 0, w/2],[0, f, h/2],[0, 0, 1]])


# TEST WITH EXISTING CAMERA
# K_temp, dist_temp = build_intrinsic_matrix(camera_intrinsics[2])
# Pv = all_camera_matrices[1]['P']
# K_virt = K_temp
# w = int(K_virt[0, 2]*2)
# h = int(K_virt[1, 2]*2)

# TEST HOMOGRAPHY 2.0
H = {}
#
for key in all_camera_matrices:
    print(key)
    K_temp, dist_temp = build_intrinsic_matrix(camera_intrinsics[key])
    H[key],center,distance = compute_homography(Pv, all_camera_matrices[key]['P'], K_virt, K_temp, plane)
    print(distance)
print('HOMO',H)# color image
color_images, stitched_image = color_virtual_image(plane, Pv, w, h, imgs, all_camera_matrices, camera_intrinsics, K_virt,'homography',H)
stitched_image = stitched_image/255
imgplot = plt.imshow(stitched_image)
plt3d = plot_3D(points3D,plane,all_camera_matrices,Pv)

# Test for visualizing the projection of a virtual pixel to the plane
pixelpoint = [0,0]
line_dir,line_point = line_from_pixel(pixelpoint,Pv,K_virt)
intersection_point = intersection_line_plane(line_dir,line_point,plane)
plt3d.scatter3D(intersection_point[0], intersection_point[1], intersection_point[2],  cmap='Blues')
plt3d.quiver(line_point[0],line_point[1],line_point[2], line_dir[0], line_dir[1], line_dir[2], length=20, color='y')
cam_center, principal_axis = get_camera_center_and_axis(Pv)
plt3d.quiver(center[0],center[1],center[2], principal_axis[0,0], principal_axis[0,1], principal_axis[0,2], length=distance, color='Red')
plt.show()

