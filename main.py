from COLMAP_functions import *
from estimate_plane import *
from color_virtual_image import *
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
print(images[1])
for key in images:
    imgs[key] = np.asarray(plt.imread(image_dir + images[key].name))
    all_camera_matrices[key] = camera_quat_to_P(images[key].qvec,images[key].tvec)

#Get all camera intrinsics
print(cameras[1])

#define virtual camera
Rv = np.eye(3)
tv = np.asarray([1, 1, 1])
Pv = np.column_stack((Rv,tv))
#define virtual image size
w = 1920
h = 1080
#color image
# color_virtual_image(plane,Pv,w,h,imgs,all_camera_matrices)


plot_3D(points3D,plane,all_camera_matrices)
# print(a,b,d)