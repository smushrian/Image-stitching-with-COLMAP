from COLMAP_functions import *
from estimate_plane import *

automatic_reconstructor()
cameras, points3D, images = get_data_from_binary()
# print(points3D[1].xyz)
a,b,d = find_plane(points3D)
print(a,b,d)

