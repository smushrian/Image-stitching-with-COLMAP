import numpy as np
from sympy import Matrix

def line_from_pixel(pixelpoint,P):
    #takes point on form [x,y]
    #P is regular camera matrix
    ray_cam = np.asarray([pixelpoint[0], pixelpoint[1], 0])
    P = Matrix(P)
    start_point = P.nullspace()[0]
    line_point = np.matmul(np.linalg.inv(P),np.asarray([pixelpoint[0],pixelpoint[1],1]))
    ray = np.matmul(np.linalg.inv(P),ray_cam)
    return ray, start_point, line_point
def intersection_line_plane(ray,ray_start,ray_point,plane):
    plane_normal = np.array([[plane[0],plane[1],plane[2]]]).T
    plane_point = np.array([[0, 0, -plane[3]/plane[2]]]).T
    ndotu = plane_normal.dot(ray)
    epsilon = 1e-6
    if abs(ndotu) < epsilon:
        print("no intersection or line is within plane")

    w = ray_point - plane_point
    si = -plane_normal.dot(w) / ndotu
    Psi = w + si * ray + plane_point

    print("intersection at", Psi)

def get_color_for_3Dpoint_in_plane(plane_point, P, image):
     X = np.asarray([plane_point[0], plane_point[1], plane_point[2], 1])
     pixels = np.matmul(P, X)
     pixels = pixels/pixels[2]
     color = image[pixels[0],pixels[1],:]
     return color


def get_color_for_virtual_pixel(image,Pvirtual,pixelpoint,plane, Preal):
    #get the ray corresponding to a pixel in virtual camera
    ray, ray_start, ray_point = line_from_pixel(pixelpoint,Pvirtual)
    #check where that ray intersects the estimated floor plane
    plane_point = intersection_line_plane(ray,ray_start,ray_point,plane)
    #get the color data from the actual image corresponding to the found 3D point in the plane
    color = get_color_for_3Dpoint_in_plane(plane_point, Preal, image)
    return color