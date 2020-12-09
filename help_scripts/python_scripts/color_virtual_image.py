import numpy as np
import math
from help_scripts.python_scripts import COLMAP_functions
from sympy import Matrix

def line_from_pixel(pixelpoint,P,K):
    #takes point on form [x,y]
    #P is regular camera matrix 3x4
    # print(K)
    # Kinv = np.linalg.inv(K)
    #normalize pixelpoints
    # print('pixels',pixelpoint)
    # norm_points = np.matmul(Kinv,np.asarray([pixelpoint[0], pixelpoint[1], 1]))
    # print('normppoints',norm_points)
    #construct ray
    # ray_cam = np.asarray([pixelpoint[0],pixelpoint[1], 1])
    # ray_cam = norm_points/norm_points[2]
    #construct matrix cam_coord -> world_coord
    # P = Matrix(P)
    # C = P.nullspace()[0]
    # R = P[:2,:2]
    # P_4x4 = np.asarray(np.vstack((P,[0, 0, 0, 1]))).astype(float)
    # print(P_4x4)
    # Pinv4x4 = np.linalg.inv(P_4x4)
    # print(Pinv4x4)
    # line_point = np.matmul(np.linalg.inv(P_4x4),np.asarray([pixelpoint[0],pixelpoint[1],1,1]))
    # line_point = [line_point[:3]/line_point[3]]
    # ray_cam_homo = np.append(ray_cam,1)
    # print(ray_cam_homo)
    # ray_global = np.matmul(np.linalg.inv(P_4x4),ray_cam_homo)
    # ray_global = ray_global[:3]/ray_global[3]
    f = K[0,0]
    cx = K[0,2]
    cy = K[1,2]
    x = pixelpoint[0]
    y = pixelpoint[1]
    R = P[0:3,0:3]
    t = P[0:3,3]
    C = -np.matmul(np.transpose(R),t)
    line_point = np.transpose(C)
    vec_len = math.sqrt(math.pow((x - cx),2) + math.pow((y - cy),2) +math.pow(f,2));
    line_dir = np.transpose((C + np.matmul(np.transpose(R),np.asarray([x-cx, y-cy, f])/vec_len))) - np.transpose(C) # unit vec
    # print('line_dir',line_dir)
    # print('line_point',line_point)

    return line_dir,line_point#, start_point, line_point
def intersection_line_plane(ray,ray_point,plane):
    plane_normal = np.array([plane[0],plane[1],plane[2]])
    plane_point = np.array([0, 0, -plane[3]/plane[2]])
    # ndotu = plane_normal.dot(ray)
    # # print('ray',ray_point)
    # epsilon = 1e-6
    # if abs(ndotu) < epsilon:
    #     print("no intersection or line is within plane")
    #     return None
    # else:
    #     w = ray_point - plane_point
    #     si = -plane_normal.dot(w) / ndotu
    #     Psi = w + si*ray + plane_point
        # print('Psi',Psi)
    # print('plane',plane)
    # print('ray', ray)
    # t = -plane[3]/(plane[0]*ray[0] + plane[1]*ray[1] + plane[2])
    t = np.divide(np.matmul((plane_point-ray_point),plane_normal),np.matmul(ray,plane_normal))
    intersection_point = ray*t
    # print("intersection at", intersection_point)
    return intersection_point

def get_color_for_3Dpoint_in_plane(plane_point, cams, images,image_w, image_h, intrinsics):
    # image_w = int(image_w)
    # image_h = int(image_h)
    colors = []
    X = np.asarray([plane_point[0], plane_point[1], plane_point[2], 1])
    for key in images:
        #get camera intrinsics
        K, dist = COLMAP_functions.build_intrinsic_matrix(intrinsics[key])
        # w = int(Ks[key][0, 2] * 2)
        # h = int(Ks[key][1, 2] * 2)
        w = len(images[key][0,:,0])
        h = len(images[key][:,0,0])
        pixelsCAM = np.matmul(cams[key]['P'], X)
        pixelsFILM = pixelsCAM/pixelsCAM[2]

        # undistort pixels
        # print(dist)
        pixels = pixelsFILM
        # pixels_dist = [0,0,1]
        pixels_dist = [pixels[0], pixels[1], 1]
        # pixels_dist[0] = pixels[0] * (1 + dist[0] * math.pow(math.sqrt(math.pow(pixels[0],2) + math.pow(pixels[1],2)),2)
        #                          + dist[1] * math.pow(math.sqrt(math.pow(pixels[0],2) + math.pow(pixels[1],2)),4))
        # pixels_dist[1] = pixels[1] * (1 + dist[0] * math.pow(math.sqrt(math.pow(pixels[0],2) + math.pow(pixels[1],2)),2)
        #                    + dist[1] * math.pow(math.sqrt(math.pow(pixels[0],2) + math.pow(pixels[1],2)),4))
        # print('distpix',pixels_dist)
        pixels = np.matmul(K,pixels_dist)
        pixels = pixels/pixels[2]

        # print('pix',(pixels))

        pix_x = int(pixels[0])
        pix_y = int(pixels[1])

        # print('x',pix_x)
        # print('y',pix_y)
        # print('w',w-1)
        # print('h',h-1)
        # print('bool', pix_x >= w - 1)
        if pix_x >= w-1 or pix_x < 0 or pix_y >= h-1 or pix_y < 0:
            color = [None, None, None]
            colors.append(color)
     # elif pixels[1] > image_h or pixels[1] > 0:
     #     color = [None, None, None]
        else:
            # print(images[key])
            # print(pixels)

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

def color_virtual_image(plane,Pvirtual,w_virtual,h_virtual,images,cams,intrinsics,K_virt):
    color_images = {1: np.zeros((h_virtual, w_virtual,  3)) , 2: np.zeros((h_virtual, w_virtual,  3)),
                    3: np.zeros((h_virtual, w_virtual,  3)), 4: np.zeros((h_virtual, w_virtual,  3))}
    stitched_image = np.zeros((h_virtual,w_virtual, 3))
    for y in range(0,h_virtual):
        # print('Loop is on: ',y)
        for x in range(0, w_virtual):
            color = get_color_for_virtual_pixel(images, Pvirtual, [x, y], plane,cams,intrinsics,w_virtual,h_virtual,K_virt)
            color_images[1][y, x, :] = color[0]
            color_images[2][y, x, :] = color[1]
            color_images[3][y, x, :] = color[2]
            color_images[4][y, x, :] = color[3]
            for i in range(0,4):
                if color[i][0] is not None:
                    stitched_image[y,x,:] = color[i]
    imgs = []
    for key in color_images:
        imgs.append(color_images[key])
    imgs = np.asarray(imgs)
    print(imgs.shape)
    return color_images, stitched_image


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


def compute_homography(P_virt, P_real, plane):
    # Assuming distance to plane to be 1.
    d = 1

    R_virt = P_virt[:, 0, 3]
    R_real = P_real[:, 0, 3]
    t_virt = P_virt[:, 3]
    t_real = P_real[:, 3]

    R_real_transpose = R_real.transpose()
    plane_norm = plane/plane[-1]
    n = plane_norm[0:3]

    H = R_virt*R_real_transpose - (-R_virt*R_real_transpose*t_real + t_virt)*n / d

    return H
