import numpy as np
import matplotlib.pyplot as plt
from help_scripts.python_scripts.COLMAP_functions import *
import cv2 as cv
import scipy.interpolate as intp


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
    coord_range = np.linspace(-0.95, 0.95, num_points, dtype=float, axis=0)

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

def filter_grid(dist_grid, undist_grid, img_size=(1280,720), margin=10):
    N = len(dist_grid)
    mask = []

    width, height = img_size

    for idx in range(N):
        if dist_grid[idx,0] >= 0-margin and dist_grid[idx,0] < width+margin and dist_grid[idx,1] >= 0-margin and dist_grid[idx,1] < height+margin:
            mask.append(True)
        else:
            mask.append(False)

    filtered_dist_grid = dist_grid[mask, :].copy()
    filtered_undist_grid = undist_grid[mask, :].copy()

    return filtered_dist_grid, filtered_undist_grid


def test_interpolator(interpolator, undist_grid, dist_grid):
    plt.figure("Visualization of undistortion")
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    intp_dist_grid = np.empty((720*1280, 2))
    intp_undist_grid = np.empty((720*1280, 2))
    nan_grid = np.empty((720 * 1280, 2))

    i = 0
    for idx in range(1280):
        for idy in range(720):
            intp_dist_grid[i, :] = np.array([idx, idy])
            intp_undist_grid[i, :] = interpolator(np.array([idx]), np.array([idy]))[0]

            if np.isnan(np.sum(intp_undist_grid[i, :])):
                nan_grid[i, :] = np.array([idx, idy])

            i = i + 1

    ax1.scatter(undist_grid[:, 0], undist_grid[:, 1], s=5, marker='.', color='r', lw=0.5, alpha=0.6)
    ax1.scatter(dist_grid[:, 0], dist_grid[:, 1], s=5, marker='.', color='b', lw=0.5, alpha=0.6)
    ax1.legend(("undistorted pixels", "distorted pixels"), loc='upper left')
    ax1.set(xlim=(-330, 1600), ylim=(-200, 900))

    ax2.scatter(intp_dist_grid[:, 0], intp_dist_grid[:, 1], s=5, marker='.', color='b', lw=0.5, alpha=0.6)
    ax2.scatter(intp_undist_grid[:, 0], intp_undist_grid[:, 1], s=5, marker='.', color='r', lw=0.5, alpha=0.6)

    ax2.scatter(nan_grid[:, 0], nan_grid[:, 1], s=5, marker='.', color='y', lw=0.5, alpha=0.6)
    ax2.legend(("interpolated, undistorted pixels", "distorted pixels", "distorted pixels interpolator returned NaN"), loc='upper left')
    ax2.set(xlim=(-330, 1600), ylim=(-200, 900))

    plt.show()


def visualize_distortion_in_pixelspace(dist_grid, undist_grid, img):
    plt.figure("Visualization of distortion in pixelspace")
    plt.imshow(img)

    plt.scatter(dist_grid[:, 0], dist_grid[:, 1], s=5, marker='.', color='b', lw=0.5, alpha=0.6)
    plt.scatter(undist_grid[:, 0], undist_grid[:, 1], s=5, marker='.', color='r', lw=0.5, alpha=0.6)
    plt.legend(("Distorted pixels", "Undistorted pixels"))

def construct_maps():

    # HYPERPARAMS
    k_list = [-0.35, -0.3, -0.2, -0.15]  # Distortion factor (set manually, not from colmap)
    num_samples = 200  # number of samples from distortion mdoel along each axis (tot # samples = num_samples**2)
    image_dir = r'/Users/ludvig/Documents/SSY226 Design project in MPSYS/Image-stitching-with-COLMAP/COLMAP_w_CUDA/'

    quick_interpolate = True
    use_full_img = False

    cameras, _, images = get_data_from_binary(image_dir)

    img_maps = {}

    img_num=1

    for cam_key in cameras.keys():

        k = k_list[img_num-1]

        cam_id = cameras[cam_key].id
        params = cameras[cam_key].params
        for img_key in images.keys():
            if images[img_key].camera_id == cam_id:
                image_name = images[img_key].name
                break

        #img_raw = cv.imread(image_dir+r'images/'+image_name)
        #img = cv.cvtColor(img_raw, cv.COLOR_BGR2RGB)

        f, cx, cy, _ = params
        K = np.array(([f, 0., cx],
                      [0,  f, cy],
                      [0,  0,  1]))
        #######################################################################

        #visualize_distortion(k)

        undistorted_grid, distorted_grid, delta_grid = sample_distortion_model(k, num_samples)

        # Transforms grid from z=1 to z=f
        undist_grid = transform_coords(K, undistorted_grid)
        dist_grid = transform_coords(K, distorted_grid)

        #dist_grid, undist_grid = filter_grid(dist_grid, undist_grid, img_size=(1280, 720), margin=10)

        #visualize_distortion_in_pixelspace(dist_grid, undist_grid, img)

        print("Fitting interpolator...", end="")
        if not quick_interpolate:
            rbf_function = 'linear'  # [multiquadric, gaussian, quintic, cubic, linear, thin_plate]
            Distortion = intp.Rbf(dist_grid[:, 0], dist_grid[:, 1], undist_grid, function=rbf_function, mode='N-D')
        else:
            Distortion = intp.CloughTocher2DInterpolator(points=undist_grid, values=dist_grid)
        print("", end="\r")

        #test_interpolator(Distortion, undist_grid, dist_grid)

        if use_full_img:
            raise NotImplementedError
            map_x = np.empty((1200, 2000), dtype=np.float32)
            map_y = np.empty((1200, 2000), dtype=np.float32)
        else:
            map_x = np.empty((720, 1280), dtype=np.float32)
            map_y = np.empty((720, 1280), dtype=np.float32)

        print("Constructing map "+str(img_num)+"/4...")
        for idx in range(map_x.shape[1]):
            for idy in range(map_x.shape[0]):

                if use_full_img:
                    interpolated_coord = Distortion(np.array([idx-250]), np.array([idy-400]))[0]
                    map_x[idy, idx] = interpolated_coord[0]
                    map_y[idy, idx] = interpolated_coord[1]
                else:
                    interpolated_coord = Distortion(np.array([idx]), np.array([idy]))[0]
                    map_x[idy, idx] = interpolated_coord[0]
                    map_y[idy, idx] = interpolated_coord[1]

            # Display progress
            print("", end="\r")
            print("{:.2f} % done.".format(100*(idx+1)/map_x.shape[1]), end="")
        print("")

        img_num = img_num+1
        img_maps[img_key] = (np.array(map_x, dtype=np.float32), np.array(map_y, dtype=np.float32))

    return img_maps


img_maps = construct_maps()

image_dir = r'/Users/ludvig/Documents/SSY226 Design project in MPSYS/Image-stitching-with-COLMAP/COLMAP_w_CUDA/'
cameras, _, images = get_data_from_binary(image_dir)

i = 1
plt.figure("test remapping")

for img_key in images.keys():

    image_name = images[img_key].name
    img_raw = cv.imread(image_dir + r'images/' + image_name)
    img = cv.cvtColor(img_raw, cv.COLOR_BGR2RGB)

    map_x, map_y = img_maps[img_key]

    img_undistorted = cv.remap(img, map_x, map_y, cv.INTER_LANCZOS4)
    plt.subplot(2, 2, i)
    plt.imshow(img_undistorted)
    i = i+1
plt.show()


