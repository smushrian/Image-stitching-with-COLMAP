from COLMAP_functions import *
from estimate_plane import *
import numpy as np
import matplotlib.pyplot as plt
import os
automatic_reconstructor()
cameras, points3D, images = get_data_from_binary()

coordinates = []
for key in points3D:
    # print(point)
    coordinates.append(points3D[key].xyz)
    # ids.append(points[i].id)
coordinates = np.asarray(coordinates)
plane,min_outlier = ransac_find_plane(coordinates,0.01)
all_camera_matrices = {}
image_names = {}
for key in images:
    image_names[key] = images[key].name
    all_camera_matrices[key] = camera_quat_to_P(images[key].qvec,images[key].tvec)

dirname = os.path.dirname(__file__)
print(dirname)
image_dir = '../COLMAP/images/'
img = plt.imread(image_dir + image_names[1])
img = np.asarray(img)
print(np.size(img,axis=2))
plot_3D(points3D,plane,all_camera_matrices)
# print(a,b,d)

