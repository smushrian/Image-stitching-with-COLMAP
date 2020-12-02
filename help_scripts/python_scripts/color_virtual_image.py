import numpy as np
import math
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
    print('line_dir',line_dir)
    print('line_point',line_point)

    return line_dir,line_point#, start_point, line_point
def intersection_line_plane(ray,ray_point,plane):
    plane_normal = np.array([plane[0],plane[1],plane[2]])
    plane_point = np.array([0, 0, -plane[3]/plane[2]])
    ndotu = plane_normal.dot(ray)
    print('ray',ray_point)
    epsilon = 1e-6
    if abs(ndotu) < epsilon:
        print("no intersection or line is within plane")
        return None
    else:
        w = ray_point - plane_point
        si = -plane_normal.dot(w) / ndotu
        print('w',w)
        print('si',si)
        print('ray',ray)
        print('planepoint',plane_point)
        Psi = w + si*ray + plane_point
        print('Psi',Psi)
    # print('plane',plane)
    # print('ray', ray)
    # t = -plane[3]/(plane[0]*ray[0] + plane[1]*ray[1] + plane[2])
    # intersection_point = ray*t
    # print("intersection at", intersection_point)
    return Psi

def get_color_for_3Dpoint_in_plane(plane_point, Ps, images,image_w, image_h):
    colors = []
    X = np.asarray([plane_point[0], plane_point[1], plane_point[2], 1])
    for key in images:
        print('X',X)
        pixels = np.matmul(Ps[key], X)
        pixels = pixels/pixels[2]
        print('pix',pixels)
        if pixels[0] > image_w or pixels[0] < 0 or pixels[1] > image_h or pixels[1] < 0:
            color = [None, None, None]
            colors.append(color)
     # elif pixels[1] > image_h or pixels[1] > 0:
     #     color = [None, None, None]
        else:
            print(images[key])
            print(pixels)
            colors.append(images[key][int(pixels[0]),int(pixels[1]),:3])

    return colors


def get_color_for_virtual_pixel(images,Pvirtual,pixelpoint,plane, Preals,Ks,w_virtual,h_virtual,K_virt):
    #get the ray corresponding to a pixel in virtual camera
    line,line_point = line_from_pixel(pixelpoint,Pvirtual,K_virt)

    #check where that ray intersects the estimated floor plane
    plane_point = intersection_line_plane(line,line_point,plane)
    #get the color data from the actual image corresponding to the found 3D point in the plane
    color = get_color_for_3Dpoint_in_plane(plane_point, Preals, images,w_virtual,h_virtual)
    return color

def color_virtual_image(plane,Pvirtual,w_virtual,h_virtual,images,cameras,K_matrices,K_virt):
    color_images = {1: np.zeros((w_virtual, h_virtual, 3)), 2: np.zeros((w_virtual, h_virtual, 3)),
                    3: np.zeros((w_virtual, h_virtual, 3)), 4: np.zeros((w_virtual, h_virtual, 3))}
    for y in range(0,h_virtual):
        print('hej')
        for x in range(0, w_virtual):
            color = get_color_for_virtual_pixel(images, Pvirtual, [x, y], plane,cameras,K_matrices,w_virtual,h_virtual,K_virt)
            color_images[1][x, y, :] = color[0]
            color_images[2][x, y, :] = color[1]
            color_images[3][x, y, :] = color[2]
            color_images[4][x, y, :] = color[3]

    imgs = []
    for key in color_images:
        imgs.append(color_images[key])
    imgs = np.asarray(imgs)
    print(imgs.shape)
    return color_images