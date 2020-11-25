
import numpy as np
import math

from sklearn import linear_model
import open3d as o3d

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def find_plane(points):
    # {1: Point3D(id=1, xyz=array([4.62656199, -0.45836651, 18.59784168]), rgb=array([12, 16, 19]),
                # error=array(2.15046715), image_ids=array([1, 2]), point2D_idxs=array([0, 1])),}
    dict_length = len(points)
    coordinates = []
    ids = []
    for i in range(1,dict_length+1):

        # print(point)
        coordinates.append(points[i].xyz)
        ids.append(points[i].id)

    xyz = np.asarray(coordinates)

    XY = xyz[:, :2]
    Z = xyz[:, 2]
    ransac = linear_model.RANSACRegressor(residual_threshold=0.3)

    ransac.fit(XY, Z)
    a, b = ransac.estimator_.coef_  # 係数
    d = ransac.estimator_.intercept_  # 切片

    return a, b, d  # Z = aX + bY + d


def angle_rotate(a, b, d):
    x = np.arange(30)
    y = np.arange(30)
    X, Y = np.meshgrid(x, y)
    Z = a * X + b * Y + d
    rad = math.atan2(Y[1][0] - Y[0][0], (Z[1][0] - Z[0][0]))
    return rad - math.pi


def show_graph(X, Y, Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z)

    plt.show()

def plot_3D(points,plane):
    dict_length = len(points)
    coordinates = []
    # ids = []
    for i in range(1, dict_length + 1):
        # print(point)
        coordinates.append(points[i].xyz)
        # ids.append(points[i].id)

    xyz = np.asarray(coordinates)
    # ax = plt.axes(projection='3d')
    a, b, d = plane
    c = 1
    x = np.linspace(-10, 10, 10)
    y = np.linspace(-10, 10, 10)

    X, Y = np.meshgrid(x, y)
    Z = (d - a * X - b * Y) / c
    plt3d = plt.figure().gca(projection='3d')
    plt3d.plot_surface(X, Y, Z, alpha=0.5)
    # plt3d.hold(True)
    print((xyz))
    plt3d.scatter3D(xyz[:,0], xyz[:,1], xyz[:,2],  cmap='Greens')


    plt.show()