import numpy as np
import math
from help_scripts.python_scripts import COLMAP_functions
from help_scripts.python_scripts import estimate_plane
import time
from sympy import Matrix

def line_from_pixel(pixelpoint,Pvirt,K):
    # tid = time.time()

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
    # elapsed = time.time() - tid
    # print('time for line_from_pixel: ', elapsed)
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
    # elapsed = time.time() - t
    # print('time for intersection_line_plane: ', elapsed)
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
        # print('distpix',pixels_dist)
        pixels = np.matmul(K,pixels_dist)
        # pixels = pixels/pixels[2]

        pix_x = int(pixels[0])
        pix_y = int(pixels[1])

        if pix_x >= w or pix_x < 0 or pix_y >= h or pix_y < 0:
            color = [None, None, None]
            colors.append(color)
        else:
            colors.append(images[key][pix_y,pix_x,:3])
    # elapsed = time.time() - t
    # print('time for get_color_for_3Dpoint_in_plane: ', elapsed)
    return colors


def get_color_for_virtual_pixel(images,Pvirtual,pixelpoint,plane, cams,intrinsics,w_virtual,h_virtual,K_virt):
    #get the ray corresponding to a pixel in virtual camera
    line,line_point = line_from_pixel(pixelpoint,Pvirtual,K_virt)
    #check where that ray intersects the estimated floor plane
    plane_point = intersection_line_plane(line,line_point,plane)
    #get the color data from the actual image corresponding to the found 3D point in the plane
    color = get_color_for_3Dpoint_in_plane(plane_point, cams, images,w_virtual,h_virtual, intrinsics)
    return color

def color_virtual_image(plane,Pvirtual,w_virtual,h_virtual,images,cams,intrinsics,K_virt,decision_variable,H):
    if decision_variable == 'ray_tracing':
        color_images = {}
        for key in images:
            color_images[key] = np.zeros((h_virtual, w_virtual,  3))
        # color_images = {1: np.zeros((h_virtual, w_virtual,  3)) , 2: np.zeros((h_virtual, w_virtual,  3)),
        #                 3: np.zeros((h_virtual, w_virtual,  3)), 4: np.zeros((h_virtual, w_virtual,  3))}
        stitched_image = np.zeros((h_virtual,w_virtual, 3))
        for y in range(0,h_virtual):
            print('Loop is on: ',y)
            for x in range(0, w_virtual):
                # t = time.time()
                color = get_color_for_virtual_pixel(images, Pvirtual, [x, y], plane,cams,intrinsics,w_virtual,h_virtual,K_virt)
                # color_images[1][y, x, :] = color[0]
                # color_images[2][y, x, :] = color[1]
                # color_images[3][y, x, :] = color[2]
                # color_images[4][y, x, :] = color[3]
                # elapsed = time.time() - t
                # print('time for get_color_for_3Dpoint_in_plane: ', elapsed)
                for i, key in enumerate(images):
                    color_images[key][y, x, :] = color[i]
                    if color[i][0] is not None:
                        stitched_image[y,x,:] = color[i]
        # imgs = []
        # for key in color_images:
        #     imgs.append(color_images[key])
        # imgs = np.asarray(imgs)
        return color_images, stitched_image

    elif decision_variable == 'homography':
        color_images = {}
        # for key in cams:

        # color_images = {1: np.zeros((h_virtual, w_virtual, 3)), 2: np.zeros((h_virtual, w_virtual, 3)),
        #                 3: np.zeros((h_virtual, w_virtual, 3)), 4: np.zeros((h_virtual, w_virtual, 3))}
        stitched_image = np.zeros((h_virtual, w_virtual, 3))
        K = {}
        dist = {}
        for key in cams:
            color_images[key] = np.zeros((h_virtual, w_virtual, 3))
            Ktemp, disttemp = COLMAP_functions.build_intrinsic_matrix(intrinsics[key])
            K[key] = Ktemp
            dist[key] = disttemp
        for y in range(0, h_virtual):
            print('Loop is on: ', y)
            for x in range(0, w_virtual):
                for index in H:
                    # print('index',index)
                    pixel = [x, y, 1]
                    # pixel_norm = np.matmul(np.linalg.inv(K_virt),np.asarray(pixel))
                    # print(pixel_norm)
                    image_point = np.matmul(H[index], pixel)
                    # image_point = np.matmul(K[index],image_point)
                    # print(image_point)
                    image_point = image_point / image_point[-1]
                    pix_x = int(image_point[0])
                    pix_y = int(image_point[1])
                    # print('pixx',pix_x)
                    # print('pixy', pix_y)
                    # print('w and h',w_virtual,h_virtual)
                    if pix_x >= w_virtual or pix_x < 0 or pix_y >= h_virtual or pix_y < 0:
                        color_images[index][y, x, :3] = [None, None, None]
                    else:
                        color_images[index][y, x, :3] = images[index][pix_y, pix_x,:3]
                    # if color_images[index][y, x, 0] is not None:
                        stitched_image[y, x, :3] = color_images[index][y, x, :3]
        return color_images, stitched_image

    else:
        print('A decision variable with either "ray_tracing" or "homography" must be passed as argument.')



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

def create_virtual_camera(camera_matrices):
    centers = {}
    axes = {}
    for index,cam in enumerate(camera_matrices):
        cam_center, principal_axis = estimate_plane.get_camera_center_and_axis(camera_matrices[cam]['P'])
        centers[index] = cam_center
        axes[index] = principal_axis
    virt_center = (centers[0]+centers[1]+centers[2]+centers[3])/4
    virt_principal_axis = (axes[0]+axes[1]+axes[2]+axes[3])/4

    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])

    xnew = np.cross(x,np.asarray(virt_principal_axis))
    normx = math.sqrt(math.pow(xnew[0][0],2) + math.pow(xnew[0][1],2) + math.pow(xnew[0][2],2))
    xnew = xnew/normx

    ynew = np.cross(y,virt_principal_axis)
    normy = math.sqrt(math.pow(ynew[0][0],2) + math.pow(ynew[0][1],2) + math.pow(ynew[0][2],2))
    ynew = ynew/normy

    Rvirt = np.asarray([[xnew[0][0], xnew[0][1], xnew[0][2]],[ynew[0][0], ynew[0][1], ynew[0][2]],
                        [virt_principal_axis[0][0],virt_principal_axis[0][1],virt_principal_axis[0][2]]],dtype='float')

    tvirt = np.asarray(np.matmul(-Rvirt,np.asarray([virt_center[0][0],virt_center[1][0],virt_center[2][0]])),dtype='float')

    Pvirt = np.column_stack((Rvirt,tvirt))

    return Pvirt

def compute_homography(P_virt, P_real, K_virt, K_real, plane,):
    # Determine rotational matrices and translation vectors:
    R_virt = P_virt[:3, :3]
    R_real = P_real[:3, :3]
    t_virt = P_virt[:3, 3]
    t_real = P_real[:3, 3]

    # Rotation matrix transpose:
    # R_real_transpose = R_real.transpose()
    R_virt_transpose = R_virt.transpose()

    # Camera center:
    cam_center = np.matmul(-np.linalg.inv(R_virt), t_virt)
    # cam_center = np.matmul(-np.linalg.inv(R_real), t_real)

    # cam_center = np.divide(cam_center, cam_center[-1])

    # Compute distance between point and plane:
    num = abs((plane[0] * cam_center[0] + plane[1] * cam_center[1] + plane[2] * cam_center[2] + plane[3]))
    den = (math.sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]))
    d = num / den

    n = np.asarray(plane[0:3])
    n_transpose = n.transpose()

    # Compute homography:
    # H = np.matmul(R_virt, R_real_transpose) - np.matmul((np.matmul(np.matmul(-R_virt, R_real_transpose), t_real) + t_virt), n_transpose) / d
    Htemp = np.matmul(R_real, R_virt_transpose) - np.matmul((np.matmul(np.matmul(-R_real, R_virt_transpose), t_virt) + t_real), n_transpose) / d
    H = np.matmul(K_real,np.matmul(Htemp,np.linalg.inv(K_virt)))
    # H=Htemp

    return H,cam_center,d
