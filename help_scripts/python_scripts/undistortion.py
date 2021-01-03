import numpy as np
import matplotlib.pyplot as plt
from help_scripts.python_scripts.COLMAP_functions import *
import cv2 as cv
import time

def dist_model(coord, k1):
    # can only be used in image space with the origin at the principal point
    x_u = coord[0]
    y_u = coord[1]

    x_d = x_u * (1 + k1 * (x_u ** 2 + y_u ** 2))
    y_d = y_u * (1 + k1 * (x_u ** 2 + y_u ** 2))

    return np.array([x_d, y_d])


def transform_coord(K, coord):

    homo_coord = np.append(coord, np.ones(1))
    transformed_coord = np.matmul(K, homo_coord)
    transformed_coord = transformed_coord / transformed_coord[2]

    return transformed_coord[:2].copy()


def generate_map(K, k, img_size=(720, 1280), margin=(250, 150), full_size_img=False):
    K_inv = np.linalg.inv(K)

    if full_size_img:
        w, h = img_size
        marg_x, marg_y = margin
        map_x = np.empty((w + 2 * marg_y, h + 2 * marg_x))
        map_y = np.empty((w + 2 * marg_y, h + 2 * marg_x))
    else:
        map_x = np.empty(img_size)
        map_y = np.empty(img_size)


    for idx in range(map_x.shape[1]):

        for idy in range(map_x.shape[0]):
            if full_size_img:
                image_coord = transform_coord(K_inv, np.array([idx-marg_x, idy-marg_y]))
            else:
                image_coord = transform_coord(K_inv, np.array([idx, idy]))

            dist_coord = dist_model(image_coord, k)
            pixel_coord = transform_coord(K, dist_coord)
            map_x[idy, idx] = pixel_coord[0]
            map_y[idy, idx] = pixel_coord[1]

        # Display progress
        print("", end="\r")
        print("{:.2f} % done.".format(100 * (idx + 1) / map_x.shape[1]), end="")

    map_x = np.array(map_x, dtype=np.float32)
    map_y = np.array(map_y, dtype=np.float32)

    return map_x, map_y


def compute_all_maps(image_dir, k_list=None, full_size_img=True):

    if k_list is None:
        # img 1 corresponds to cam 2
        # img 2 corresponds to cam 1
        # img 3 corresponds to cam 3
        # img 4 corresponds to cam 4
        k_list = [-0.28, -0.28, -0.18, -0.15]

    cameras, _, images = get_data_from_binary(image_dir)

    map_num = 1
    maps = {}

    for cam_key in cameras.keys():
        print("Constructing map " + str(map_num) + "/4...")
        cam_id = cameras[cam_key].id
        params = cameras[cam_key].params
        for img_key in images.keys():
            if images[img_key].camera_id == cam_id:
                break  # this will make img_key to be set correctly for maps indexing

        f, cx, cy, _ = params
        K = np.array(([f, 0., cx],
                      [0, f, cy],
                      [0, 0, 1]))
        k = k_list[map_num-1]

        img_size = (720, 1280)
        margin = (250, 150)
        maps[img_key] = generate_map(K, k, img_size, margin, full_size_img)

        map_num = map_num+1
        print("")

    return maps


if __name__ == "__main__":

    image_dir = r'/Users/ludvig/Documents/SSY226 Design project in MPSYS/Image-stitching-with-COLMAP/COLMAP_w_CUDA/'

    start = time.process_time()
    maps = compute_all_maps(image_dir)
    end = time.process_time()

    print("\nConstruction of all maps took {:.2f} seconds.".format(end - start))

    _, _, images = get_data_from_binary(image_dir)

    plt.figure("Remapped images")
    i = 1
    time_sum = 0

    for img_key in images.keys():
        image_name = images[img_key].name

        img_raw = cv.imread(image_dir + r'images/' + image_name)
        img = cv.cvtColor(img_raw, cv.COLOR_BGR2RGB)

        map_x, map_y = maps[img_key]

        start = time.process_time()
        img_undistorted = cv.remap(img, map_x, map_y, cv.INTER_LANCZOS4)
        end = time.process_time()

        time_sum = time_sum + end - start

        plt.subplot(2, 2, i)
        plt.title("img key "+str(img_key))
        plt.imshow(img_undistorted)
        i = i+1

    print("Image remapping took on average {:.2f} milliseconds. (for remapping 4 images individually)".format(1000*time_sum/4))
    plt.show()



