from COLMAP_functions import *
from estimate_plane import *
import numpy as np
automatic_reconstructor()
cameras, points3D, images = get_data_from_binary()
# print(points3D[1].xyz)
# print((points3D))
# a,b,d = find_plane(points3D)
coordinates = []
# ids = []
# for i in range(1, dict_length + 1):
for key in points3D:
    # print(point)
    coordinates.append(points3D[key].xyz)
    # ids.append(points[i].id)
coordinates = np.asarray(coordinates)
print(len(coordinates))
plane,min_outlier = ransac_find_plane(coordinates,0.01)
plot_3D(points3D,plane)
# print(a,b,d)

