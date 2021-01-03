import numpy as np
import math
from help_scripts.python_scripts import COLMAP_functions
import cv2 as cv
import matplotlib.pyplot as plt

def line_from_pixel(pixelpoint,Pvirt,K):

    f = K[0,0]
    cx = K[0,2]
    cy = K[1,2]
    x = pixelpoint[0]
    y = pixelpoint[1]
    R = Pvirt[0:3, 0:3]
    t = Pvirt[0:3, 3]
    C = -np.matmul(np.transpose(R), t)
    line_point = np.transpose(C)
    vec_len = np.sqrt(np.power((x - cx), 2) + np.power((y - cy), 2) + np.power(f, 2))
    line_dir = np.transpose((C + np.matmul(np.transpose(R), np.asarray([x-cx, y-cy, f])/vec_len))) - np.transpose(C) # unit vec

    return line_dir, line_point

def intersection_line_plane(ray,ray_point,plane):
    # t = time.time()
    plane_normal = np.array([plane[0],plane[1],plane[2]])
    plane_point = np.array([0, 0, -plane[3]/plane[2]])
    ndotu = plane_normal.dot(ray)
    epsilon = 1e-6
    if abs(ndotu) < epsilon:
        print("no intersection or line is within plane")
        return None
    else:
        w = ray_point - plane_point
        si = -plane_normal.dot(w) / ndotu
        Psi = w + si*ray + plane_point

    return Psi

def get_color_for_3Dpoint_in_plane(plane_point, cams, images,image_w, image_h, intrinsics):
    # t = time.time()
    colors = []
    X = np.asarray([plane_point[0], plane_point[1], plane_point[2], 1])
    for key in images:
        #get camera intrinsics
        K, dist = COLMAP_functions.build_intrinsic_matrix(intrinsics[key])
        w = len(images[key][0,:,0])
        h = len(images[key][:,0,0])
        pixelsCAM = np.matmul(cams[key]['P'], X)
        pixelsFILM = pixelsCAM/pixelsCAM[2]

        # undistort pixels
        pixels = pixelsFILM
        # pixels_dist = [0,0,1]
        pixels_dist = [pixels[0], pixels[1],1]
        # pixels_dist[0] = pixels[0] * (1 + dist[0] * math.pow(math.sqrt(math.pow(pixels[0],2) + math.pow(pixels[1],2)),2)
        #                          + dist[1] * math.pow(math.sqrt(math.pow(pixels[0],2) + math.pow(pixels[1],2)),4))
        # pixels_dist[1] = pixels[1] * (1 + dist[0] * math.pow(math.sqrt(math.pow(pixels[0],2) + math.pow(pixels[1],2)),2)
        #                    + dist[1] * math.pow(math.sqrt(math.pow(pixels[0],2) + math.pow(pixels[1],2)),4))
        pixels_dist[0] =  pixels[0] * (1 + dist[0]*(math.pow(pixels[0],2)+math.pow(pixels[1],2)))
        pixels_dist[1] = pixels[1] * (1 + dist[0] * (math.pow(pixels[0],2) + math.pow(pixels[1],2)))
        pixels = np.matmul(K,pixels_dist)
        pix_x = int(pixels[0])
        pix_y = int(pixels[1])

        if pix_x >= w or pix_x < 0 or pix_y >= h or pix_y < 0:
            color = [None, None, None]
            colors.append(color)
        else:
            colors.append(images[key][pix_y,pix_x,:3])

    return colors


def get_color_for_virtual_pixel(images,Pvirtual,pixelpoint,plane, cams,intrinsics,w_virtual,h_virtual,K_virt):
    #get the ray corresponding to a pixel in virtual camera
    line,line_point = line_from_pixel(pixelpoint,Pvirtual,K_virt)
    #check where that ray intersects the estimated floor plane
    plane_point = intersection_line_plane(line,line_point,plane)
    #get the color data from the actual image corresponding to the found 3D point in the plane
    color = get_color_for_3Dpoint_in_plane(plane_point, cams, images,w_virtual,h_virtual, intrinsics)
    return color


def stitch_image(plane,Pvirtual,w_virtual,h_virtual,images,cams,intrinsics,K_virt,decision_variable,H):

    stitched_image = np.zeros((h_virtual, w_virtual, 3))

    if decision_variable == 'ray_tracing':
        print('\nStitching image using ray tracing...')
        color_images = {}
        for key in images:
            color_images[key] = np.zeros((h_virtual, w_virtual,  3))

        for y in range(0, h_virtual):

            for x in range(0, w_virtual):
                color = get_color_for_virtual_pixel(images, Pvirtual, [x, y], plane,cams,intrinsics,w_virtual,h_virtual,K_virt)
                for i, key in enumerate(images):
                    color_images[key][y, x, :] = color[i]
                    if color[i][0] is not None:
                        stitched_image[y,x,:] = color[i]

            print("", end="\r")
            print("{:.2f} % done.".format(100 * (y + 1) / h_virtual), end="")

        return color_images, stitched_image

    elif decision_variable == 'homography':
        print('\nStitching image using homography...')

        color_images = {}

        K_virt_inv = np.linalg.inv(K_virt)

        K = {}
        #dist = {}
        w_real = {}
        h_real = {}

        for key in range(1, 5):
            #color_images[key] = np.zeros((h_virtual, w_virtual, 3))
            K[key], _ = COLMAP_functions.build_intrinsic_matrix(intrinsics[key])

            #dist[key] = disttemp
            w_real[key] = len(images[key][0, :, 0])
            h_real[key] = len(images[key][:, 0, 0])

        homo_pixels = np.stack((np.tile(np.arange(w_virtual), h_virtual),
                                np.repeat(np.arange(h_virtual), w_virtual, axis=0),
                                np.ones((w_virtual*h_virtual,))), axis=1).T

        pixels_norm = np.matmul(K_virt_inv, homo_pixels)

        for index in range(1, 5):

            img_pts = np.matmul(H[index], pixels_norm)
            img_pts = np.matmul(K[index], img_pts/img_pts[2, :])[0:2, :].astype(int)

            for idx in range(img_pts.shape[1]):

                if img_pts[0, idx] < w_real[index] and img_pts[0, idx] >= 0 and img_pts[1, idx] < h_real[index] and img_pts[1, idx] >= 0:
                    stitched_image[idx//h_virtual, idx % w_virtual, :] = images[index][img_pts[1, idx], img_pts[0, idx], :3]

        print("100.00 % done.")
        return color_images, stitched_image/255  # /255 to convert colors

    else:
        print('A decision variable with either "ray_tracing" or "homography" must be passed as argument.')


def compute_homography(P_virt, P_real, K_virt, K_real, plane):
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
    H = P_real_trans[0:3,0:3] - (t_real_trans@n)/(plane_new[3])
    return H,plane_new,P_real_trans,P_virt_trans

"""
# Currently unused
def mean_color(color_images, w_virtual, h_virtual):
    mean_color_matrix = np.zeros((h_virtual, w_virtual, 3))

    c_im1 = color_images[1]
    c_im2 = color_images[2]
    c_im3 = color_images[3]
    c_im4 = color_images[4]

    # c_im = [c_im1, c_im2, c_im3, c_im4]

    for y in range(0, h_virtual):
        for x in range(0, w_virtual):
            im_arr = [c_im1[y, x, :], c_im2[y, x, :], c_im3[y, x, :], c_im4[y, x, :]]
            i = 0
            for c in im_arr:
                c_sum = np.sum(c)

                if np.isnan(c_sum):
                    im_arr.pop(i)
                    i = 0
                else:
                    i += 1

            # mean_col = np.mean([c_im1[y, x, :], c_im2[y, x, :], c_im3[y, x, :], c_im4[y, x, :]], axis=0)
            mean_col = np.mean(im_arr, axis=0)
            print(mean_col)
            mean_color_matrix[y, x] = mean_col

    return mean_color_matrix
"""