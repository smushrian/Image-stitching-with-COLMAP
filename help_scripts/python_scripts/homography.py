import numpy as np
import math
from help_scripts.python_scripts import COLMAP_functions
import cv2 as cv
import matplotlib.pyplot as plt


def compute_stitching_map(w_virtual, h_virtual, images, intrinsics, K_virt, decision_variable,H):

    if decision_variable == 'homography':
        print('\nComputing stitching map...')

        K_virt_inv = np.linalg.inv(K_virt)

        K = {}
        # dist = {}
        w_real = {}
        h_real = {}

        for key in range(1, 5):
            # color_images[key] = np.zeros((h_virtual, w_virtual, 3))
            K[key], _ = COLMAP_functions.build_intrinsic_matrix(intrinsics[key])

            # dist[key] = disttemp
            w_real[key] = len(images[key][0, :, 0])
            h_real[key] = len(images[key][:, 0, 0])

        homo_pixels = np.stack((np.tile(np.arange(w_virtual), h_virtual),
                                np.repeat(np.arange(h_virtual), w_virtual, axis=0),
                                np.ones((w_virtual * h_virtual,))), axis=1).T

        pixels_norm = np.matmul(K_virt_inv, homo_pixels)

        img_pts = {}
        for index in range(1, 5):

            img_pts[index] = np.matmul(H[index], pixels_norm)
            img_pts[index] = np.matmul(K[index], img_pts[index] / img_pts[index][2, :])[0:2, :].astype(int)

        print("100.00 % done.")
        return img_pts


def stitch_image_with_map(images, img_pts, w_virtual, h_virtual, w_real=1280, h_real=720):

    print('\nStitching image...')
    stitched_image = np.zeros((h_virtual, w_virtual, 3))

    for index in range(1, 5):
        for idx in range(img_pts[index].shape[1]):

            if w_real > img_pts[index][0, idx] >= 0 and \
                    h_real > img_pts[index][1, idx] >= 0:
                stitched_image[idx//h_virtual, idx % w_virtual, :] = images[index][img_pts[index][1, idx],
                                                                                   img_pts[index][0, idx], :3]

    print("100.00 % done.")
    return stitched_image/255  # /255 to convert colors


def compute_homography_matrix(P_virt, P_real, K_virt, K_real, plane):
    # Determine rotational matrices and translation vectors:
    #print('plane: ',plane)
    #print('pvirt: ',P_virt)
    #print('preal: ', P_real)
    #print(P_virt)

    #create transform
    P_transform = np.vstack((P_virt,[0, 0, 0, 1]))

    #transform cameras
    P_real_trans = np.matmul(P_real,np.linalg.inv(P_transform))
    P_virt_trans = np.matmul(P_virt,np.linalg.inv(P_transform))

    #print('det: ',np.linalg.det(P_real_trans[:3,:3]))

    #create new plane variable so old one isnt changed
    normed_plane = plane[:]
    #normalize plane
    normed_plane = normed_plane / np.linalg.norm(normed_plane[:3])

    #create point on plane and transform it to calculate d'
    point_on_plane = normed_plane[0:3]*normed_plane[3]
    point_on_plane = np.asarray([point_on_plane[0],point_on_plane[1],point_on_plane[2],1])
    point_on_plane = np.matmul(P_transform,point_on_plane)

    #transform normal to plane (a' b' c')
    n_prim = np.asarray([normed_plane[0],normed_plane[1],normed_plane[2],0])
    n_prim = np.matmul(np.transpose(np.linalg.inv(P_transform)),n_prim)

    #calculate d'
    d_prim = np.dot(point_on_plane,n_prim)
    #put together new plane
    plane_new = np.asarray([n_prim[0],n_prim[1],n_prim[2],d_prim])

    #fix vectors for proper vector multiplication when calculating H
    t_real_trans = P_real_trans[0:3,3].reshape(3,1)
    n = plane_new[0:3].reshape(1,3)
    #calculate H
    H = P_real_trans[0:3, 0:3] - (t_real_trans@n)/(plane_new[3])

    return H, plane_new, P_real_trans, P_virt_trans
