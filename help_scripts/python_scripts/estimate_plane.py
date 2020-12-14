import numpy as np
import math
import matplotlib.pyplot as plt
from sympy import Matrix
from scipy.spatial.transform import Rotation as R


def plot_3D(points,plane,all_cameras,cam_virt):
    dict_length = len(points)
    coordinates = []
    # ids = []
    # for i in range(1, dict_length + 1):
    for key in points:
        # print(point)
        coordinates.append(points[key].xyz)
        # ids.append(points[i].id)

    xyz = np.asarray(coordinates)
    # ax = plt.axes(projection='3d')
    a, b, c, d = plane
    x = np.linspace(-5, 5, 10)
    y = np.linspace(-5, 5, 10)

    X, Y = np.meshgrid(x, y)
    Z = (d + a * X + b * Y) / -c
    plt3d = plt.figure().gca(projection='3d')
    plt3d.plot_surface(X, Y, Z, alpha=0.5)
    # plt3d.hold(True)
    plt3d.scatter3D(xyz[:,0], xyz[:,1], xyz[:,2],  cmap='Greens')
    for key in all_cameras:
        cam_center, principal_axis = get_camera_center_and_axis(all_cameras[key]['P'])
        plt3d.quiver(cam_center[0,0],cam_center[1,0],cam_center[2,0], principal_axis[0,0], principal_axis[0,1], principal_axis[0,2], length=2, color='r')
    cam_center_virt, principal_axis_virt = get_camera_center_and_axis(cam_virt)
    plt3d.quiver(cam_center_virt[0,0],cam_center_virt[1,0],cam_center_virt[2,0], principal_axis_virt[0,0], principal_axis_virt[0,1], principal_axis_virt[0,2], length=2, color='b')
    #plt.show()
    return plt3d

def get_camera_center_and_axis(P):
    P = Matrix(P)
    cam_center = P.nullspace()[0]
    principal_axis = P[2, :3]
    return np.asarray(cam_center), np.asarray(principal_axis)


def ransac_find_plane(pts, threshold):
# pts: Nx3, N 3D points
# threshold: Scalar, threshold , threshold for points to be inliers
# plane: 4x1, plane on the form ax + by + cz + d = 0

    if threshold == 0:
        print('Threshold = 0 may give false outliers due to machine precision errors')

    # Init
    N = len(pts)
    epsilon = 0.4  # epsilon0

    mismatch_prob = 0.3  # eta

    kmax = math.log(mismatch_prob) / math.log(1 - math.pow(epsilon, 3))
    min_outliers = N
    k = 1

    while k < kmax:
        #Select subset of points and calculate preliminary plane
        subset = np.random.randint(0, N, 4)  # randomize 3 points
        pts_prim = pts[subset, :]

        plane_prel = compute_plane(pts_prim)

        #Measure performance of prel plane
        residual_lengths = residual_lengths_points_to_plane(pts, plane_prel)

        outliers = sum(residual_lengths > threshold)
        inliers = N - outliers

        # Log keeping
        if (outliers > 0):
            if outliers < min_outliers:
                # if best sub-perfect case found, save loss and iterate
                min_outliers = outliers

                plane = plane_prel

                epsilon = inliers / N
                kmax = math.log(mismatch_prob) / math.log(1 - math.pow(epsilon,3))
        else:
            # If best case found(outliers=0), return immidiately
            print('# of outliers !> 0. (this case is not yet tested)')
            plane = plane_prel
            min_outliers = outliers
            print('Total # of iterations was ' + str(k) + ' with 0% outliers.')
            return plane, min_outliers

        k = k + 1

    print('Total # of iterations was ' + str(k) + ' and optimal percentage of outliers was ' + str((min_outliers / N) * 100) + '%.')
    return plane, min_outliers

def compute_plane(pts):
    # pts: 3x3, 3 3D points of form 3x1
    # plane: 4x1, plane such that Ax + By + Cz + D = 0

    A = pts[0, :]
    B = pts[1, :]
    C = pts[2, :]

    AB = B - A
    AC = C - A

    N = np.cross(AB, AC)

    plane = [N[0], N[1], N[2], -(N[0]*A[0] + N[1]*A[1] + N[2]*A[2])]  # sum(np.multiply(-C, N))]
    return plane

def residual_lengths_points_to_plane(pts, plane):
    #pts: 3xN 3D points
    #plane: 4x1 [a,b,c,d] such that ax+by+cz+d=0
    #residual_lengths: 1xN the minimum distance from all points to the plane
    N = len(pts[:, 0])
    normal_vec = np.divide(plane[:3], math.sqrt(sum(np.power(plane[:3], 2))))
    residual_lengths = np.zeros(N)

    # find a point on the plane
    if plane[0] != 0:
        P = [-plane[3]/plane[0], 0, 0]
    elif plane[1] != 0:
        P = [0, -plane[3] / plane[1], 0]
    elif plane[2] != 0:
        P = [0, 0, -plane[3] / plane[2]]
    else:
        P = [0, 0, 0]

    for i in range(0, N):
        #difference vector from plane to point
        u = pts[i, :] - P

        #length of difference vector projected onto normal vector
        residual_lengths[i] = abs(np.dot(u, normal_vec))

    return residual_lengths

def camera_quat_to_P(quat, t):
    quat_scalar_last = [quat[1],quat[2],quat[3],quat[0]]
    R_matrix = R.from_quat(quat_scalar_last).as_matrix()
    t = np.asarray(t)
    P = np.column_stack((R_matrix,t))
    cam = {'P': P, 'R': R_matrix, 't': t}
    return cam