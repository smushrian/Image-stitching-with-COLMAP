import numpy as np
import matplotlib.pyplot as plt
from help_scripts.python_scripts.COLMAP_functions import *
import cv2 as cv
import scipy.interpolate as intp  # interpolation


def visualize_distortion(k):

    undist_pts, dist_pts, delta_pts = sample_distortion_model(k, 15)

    plt.figure("1")
    plt.title('Relationship between un-/distorted coordinates at z=1')

    plt.plot(undist_pts[:, 0], undist_pts[:, 1], 'ro', markersize=4)
    plt.quiver(undist_pts[:, 0], undist_pts[:, 1], delta_pts[:, 0], delta_pts[:, 1], units='xy', scale=1.0, scale_units='xy')
    plt.plot(dist_pts[:, 0], dist_pts[:, 1], 'bo', markersize=4)

    plt.legend(("Undistorted point", "Distorted point"))
    plt.show()


def dist_model(coord, k1):
    # These coordinates have their origin at the camera principal point
    x_u = coord[0]
    y_u = coord[1]

    x_d = x_u * (1 + k1 * (x_u ** 2 + y_u ** 2))
    y_d = y_u * (1 + k1 * (x_u ** 2 + y_u ** 2))

    return np.array([x_d, y_d])


def sample_distortion_model(k, num_points):
    """
    :param k: [float] distortion factor
    :param num_points: [int] sqrt(# total of samples from model)
    :return undistorted_points: [numpy array] [num_points**2,2] [x_u, y_u] equidistant, undistorted points
    :return distorted_points: [numpy array] [num_points**2,2] [x_d, y_d] distorted points, equidistant, undistorted
                                                                         points propagated through distortion model
    :return delta_points: [numpy array] [num_points**2,2] [x_d-x_u, y_d-y_u]
    """
    coord_range = np.linspace(-0.9, 0.9, num_points, dtype=float, axis=0)

    undistorted_points = np.empty((num_points**2, 2))
    distorted_points = np.empty((num_points**2, 2))
    delta_points = np.empty((num_points**2, 2))

    for x_idx, x_u in enumerate(coord_range):
        for y_idx, y_u in enumerate(coord_range*0.6):

            coord_u = np.array([x_u, y_u])
            coord_d = dist_model(coord_u, k)
            delta = coord_d - coord_u

            undistorted_points[x_idx * num_points + y_idx, :] = coord_u
            distorted_points[x_idx * num_points + y_idx, :] = coord_d
            delta_points[x_idx * num_points + y_idx, :] = delta

    return undistorted_points, distorted_points, delta_points

def transform_coords(K, coords):
    N = len(coords)
    transformed_coords = np.empty((N, 3))

    homo_coords = np.append(coords, np.ones((N, 1)), axis=1)

    for idx in range(N):
        transformed_coords[idx, :] = np.matmul(K, homo_coords[idx, :])
        transformed_coords[idx, :] = transformed_coords[idx, :] / transformed_coords[idx, 2]

    return transformed_coords[:, :2]

def filter_grid(dist_grid, undist_grid, img_size=(1280,720)):
    N = len(dist_grid)
    mask = []

    width, height = img_size

    for idx in range(N):
        if dist_grid[idx,0] >= 0 and dist_grid[idx,0] < width and dist_grid[idx,1] >= 0 and dist_grid[idx,1] < height:
            mask.append(True)
        else:
            mask.append(False)

    filtered_dist_grid = dist_grid[mask,:].copy()
    filtered_undist_grid = undist_grid[mask,:].copy()

    #plt.figure()
    #plt.imshow(img)
    #plt.scatter(filtered_dist_grid[:, 0], filtered_dist_grid[:, 1], s=5, marker='.', color='b', lw=0.5, alpha=0.6)
    #plt.scatter(filtered_undist_grid[:, 0], filtered_undist_grid[:, 1], s=5, marker='.', color='r', lw=0.5, alpha=0.6)
    #plt.show()

    return filtered_dist_grid, filtered_undist_grid

def update_map(ind, map_x, map_y):
    if ind == 0:
        for i in range(map_x.shape[0]):
            for j in range(map_x.shape[1]):
                if j > map_x.shape[1]*0.25 and j < map_x.shape[1]*0.75 and i > map_x.shape[0]*0.25 and i < map_x.shape[0]*0.75:
                    map_x[i,j] = 2 * (j-map_x.shape[1]*0.25) + 0.5
                    map_y[i,j] = 2 * (i-map_y.shape[0]*0.25) + 0.5
                else:
                    map_x[i,j] = 0
                    map_y[i,j] = 0

    elif ind == 1:
        for i in range(map_x.shape[0]):
            map_x[i,:] = [x for x in range(map_x.shape[1])]
        for j in range(map_y.shape[1]):
            map_y[:,j] = [map_y.shape[0]-y for y in range(map_y.shape[0])]

    elif ind == 2:
        for i in range(map_x.shape[0]):
            map_x[i,:] = [map_x.shape[1]-x for x in range(map_x.shape[1])]
        for j in range(map_y.shape[1]):
            map_y[:,j] = [y for y in range(map_y.shape[0])]

    elif ind == 3:
        for i in range(map_x.shape[0]):
            map_x[i,:] = [map_x.shape[1]-x for x in range(map_x.shape[1])]
        for j in range(map_y.shape[1]):
            map_y[:,j] = [map_y.shape[0]-y for y in range(map_y.shape[0])]






# HYPERPARAMS
k = -0.23  # Distortion factor (set manually, not from colmap)
num_samples = 140  # number of sampels from distortion mdoel along each axis (tot # samples = num_samples**2)
image_dir = r'/Users/ludvig/Documents/SSY226 Design project in MPSYS/Image-stitching-with-COLMAP/COLMAP_w_CUDA/'

#######################################################################
# to ensure the correct params (K) is paired with the correct image
cameras, _, images = get_data_from_binary(image_dir)
cam_id = cameras[1].id
params = cameras[1].params
for key in images.keys():
    if images[key].camera_id == cam_id:
        image_name = images[key].name
        break

img_raw = cv.imread(image_dir+r'images/'+image_name)
img = cv.cvtColor(img_raw, cv.COLOR_BGR2RGB)

f, cx, cy, _ = params
K = np.array(([f, 0., cx],
              [0,  f, cy],
              [0,  0,  1]))

plt.figure()
plt.imshow(img)
#######################################################################

#visualize_distortion(k)

# Sample distortion model
undistorted_grid, distorted_grid, delta_grid = sample_distortion_model(k, num_samples)

# Detta transformerar griddet från "image space" till "pixel space" mha multiplicering med K
undist_grid = transform_coords(K, undistorted_grid)
dist_grid = transform_coords(K, distorted_grid)

# Idk om det behövs men detta filtrerar bort punkter som är utanför source bilden
filt_dist_grid, filt_undist_grid = filter_grid(dist_grid, undist_grid, img_size=(1280, 720))

# Overlayar punkterna med den faktiskt "rå"bilden
plt.scatter(filt_dist_grid[:, 0], filt_dist_grid[:, 1], s=5, marker='.', color='b', lw=0.5, alpha=0.6)
plt.scatter(filt_undist_grid[:, 0], filt_undist_grid[:, 1], s=5, marker='.', color='r', lw=0.5, alpha=0.6)

# OpenCV vill kolla för varje pixel i "destinationsbilden" (dvs den undistortade bilden), var den kommer ifrån i "sourcebilden"
# därför behöver vi använda en interpolator för att kunna leta efter heltalsvärden
quick_interpolate = True  # quick funkar bra men slutgiltliga borde göras med rbf
if not quick_interpolate:
    rbf_function = 'linear'  # [multiquadric, gaussian, quintic, cubic, linear, thin_plate]
    Distortion = intp.Rbf(filt_undist_grid[:, 0], filt_undist_grid[:, 1], filt_dist_grid, function = rbf_function, mode='N-D')
else:
    Distortion = intp.CloughTocher2DInterpolator(points=filt_undist_grid, values=filt_dist_grid)

# initierar maps, används dessa som de är med remap så kommer inte bilden att ändras!
map_x = np.array(np.repeat(np.arange(img.shape[1]).reshape((-1, 1)), img.shape[0], axis=1).T, dtype=np.float32)
map_y = np.array(np.repeat(np.arange(img.shape[0]).reshape((-1, 1)), img.shape[1], axis=1), dtype=np.float32)

# loopa x och y index för att se var remap sedan ska ta pixel värdet ifrån i source bilden
for idx in range(map_x.shape[0]):
    for idy in range(map_x.shape[1]):

        lookup = Distortion(np.array([idx-198]), np.array([idy-117]))[0]
        map_x[idx, idy] = lookup[1]
        map_y[idx, idy] = lookup[0]

# konvertering för remap
map_x = np.array(map_x, dtype=np.float32)
map_y = np.array(map_y, dtype=np.float32)

# remappning
img_undistorted = cv.remap(img, map_x, map_y, cv.INTER_LANCZOS4)

fig = plt.figure()
plt.imshow(img_undistorted)
plt.show()




