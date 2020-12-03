from help_scripts.python_scripts.COLMAP_functions import *
from help_scripts.python_scripts.estimate_plane import *
from help_scripts.python_scripts.color_virtual_image import *
import numpy as np
import matplotlib.pyplot as plt
import os

#Perform the reconstruction to get data
automatic_reconstructor()
cameras, points3D, images = get_data_from_binary()

coordinates = []
for key in points3D:
    # print(point)
    coordinates.append(points3D[key].xyz)
    # ids.append(points[i].id)
coordinates = np.asarray(coordinates)

#Estimate a floor plane
plane,min_outlier = ransac_find_plane(coordinates,0.3)


#Get all camera matrices
all_camera_matrices = {}
imgs = {}
image_dir = '../COLMAP/images/'
for key in images:
    imgs[key] = np.asarray(plt.imread(image_dir + images[key].name))
    all_camera_matrices[key] = camera_quat_to_P(images[key].qvec,images[key].tvec)

#Get all camera intrinsics
# print(cameras[1])
# K_matrices = {}
# for key in cameras:
#     K_matrices[key] = build_intrinsic_matrix(cameras[key])
#     print(cameras[key].params)

#calculate a point in the middle of all cameras
# c1 =
# for key in all_camera_matrices:

#define virtual camera
# Rv = np.eye(3)
# tv = np.asarray([1, 1, 1])
# Pv = np.column_stack((Rv,tv))
# #define virtual image size
# w = 480
# h = 360
# f = 1
# K_virt = np.asarray([[f, 0, w/2],[0, f, h/2],[0, 0, 1]])
#POSSIBLE VIRT CAMERA CENTER:
# Rv = np.asarray([[ 0.7810,-0.0492,-0.6226],
#          [-0.0057, 0.9963,-0.0858],
#           [0.6245, 0.0706, 0.7778]])
#
# tv = np.asarray([-1.5546, -1.3919, 2.1106])
# Pv = np.column_stack((Rv,tv))
# w = 1280
# h = 720
# f = 1
# K_virt = np.asarray([[f, 0, w/2],[0, f, h/2],[0, 0, 1]])
#TEST WITH EXISTING CAMERA
K_temp,dist_temp = build_intrinsic_matrix(cameras[1])
Pv = all_camera_matrices[1]['P']
K_virt = K_temp
w = int(K_virt[0,2]*2)
h = int(K_virt[1,2]*2)
#color image
color_images,stitched_image = color_virtual_image(plane,Pv,w,h,imgs,all_camera_matrices,cameras,K_virt)
print(color_images[1])
stitched_image = stitched_image/255
# plt.figure(1)
imgplot = plt.imshow(stitched_image)
plot_3D(points3D,plane,all_camera_matrices)
# print(a,b,d)