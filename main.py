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
print(points3D)
coordinates = []
for key in points3D:
    coordinates.append(points3D[key].xyz)
coordinates = np.asarray(coordinates)

#Estimate a floor plane
plane, min_outlier = ransac_find_plane(coordinates, 0.3)

# Get all camera matrices and images
all_camera_matrices = {}
imgs = {}
image_dir = '../COLMAP_w_CUDA/images/'
for key in images:
    print(images[key].camera_id)
    imgs[key] = np.asarray(plt.imread(image_dir + images[key].name))
    all_camera_matrices[images[key].camera_id] = camera_quat_to_P(images[key].qvec, images[key].tvec)

# POSSIBLE VIRT CAMERA CENTER:
# Pv = create_virtual_camera(all_camera_matrices)
# # # print(Pv)
# w = 100
# h = 100
# f = 75
# K_virt = np.asarray([[f, 0, w/2],[0, f, h/2],[0, 0, 1]])

# TEST WITH EXISTING CAMERA
K_temp, dist_temp = build_intrinsic_matrix(cameras[2])
Pv = all_camera_matrices[2]['P']
K_virt = K_temp
w = int(K_virt[0, 2]*2)
h = int(K_virt[1, 2]*2)

# TEST HOMOGRAPHY 2.0
H = {}

for key in all_camera_matrices:
    H[key] = compute_homography(Pv, all_camera_matrices[key]['P'], plane)

# print('Homography: ', H)
# H=1
# color image
color_images, stitched_image = color_virtual_image(plane, Pv, w, h, imgs, all_camera_matrices, cameras, K_virt,'homography',H)
stitched_image = stitched_image/255
imgplot = plt.imshow(stitched_image)
plt3d = plot_3D(points3D,plane,all_camera_matrices,Pv)


# Test for visualizing the projection of a virtual pixel to the plane
# pixelpoint = [0,0]
# line_dir,line_point = line_from_pixel(pixelpoint,Pv,K_virt)
# intersection_point = intersection_line_plane(line_dir,line_point,plane)
# plt3d.scatter3D(intersection_point[0], intersection_point[1], intersection_point[2],  cmap='Blues')
# plt3d.quiver(line_point[0],line_point[1],line_point[2], line_dir[0], line_dir[1], line_dir[2], length=20, color='y')
plt.show()

