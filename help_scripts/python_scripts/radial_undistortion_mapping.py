from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt


def visualize_radial_model(k):

    idx = 0
    plt.figure("2")

    for x_d in [-1, 0, 1]:

        for y_d in [-1, 0, 1]:
            idx = idx + 1
            plt.subplot(3, 3, idx)
            points = np.linspace(-3, 3)
            func = lambda undistorted_coord: radial_distortion_model(undistorted_coord, [x_d, y_d], k)

            plt.title("Evaluated at: " + str([x_d, y_d]))

            dist = func((x_d*points, y_d*points))
            sol = fsolve(func, np.array([x_d, y_d]))

            plt.plot(0, 0, 'ro')
            plt.plot(sol[0], sol[1], 'bo')

            plt.plot(dist[0, :], dist[1, :])

            plt.legend(("true sol (0,0)", "fsolve's sol"))

    plt.show()


def visualize_distortion(undist_dict, k1, img_size):
    x_range = np.linspace(-1, 1, 10)
    y_range = np.linspace(-1, 1, 10)

    plt.figure("1")
    plt.subplot(1, 2, 1)
    plt.title('Distortion introduced by lens')
    for x_u in x_range:
        for y_u in y_range:
            x_d = x_u * (1 + k1 * (x_u**2 + y_u**2))
            y_d = y_u * (1 + k1 * (x_u**2 + y_u**2))

            dx = x_d - x_u
            dy = y_d - y_u

            plt.plot(x_d, y_d, 'bo', markersize=4)
            plt.plot(x_u, y_u, 'ro', markersize=4)
            plt.quiver(x_u, y_u, dx, dy, angles='xy', scale_units='xy', scale=1, width=0.005, headwidth=3)
    plt.legend(("Distorted point", "Undistorted point"))

    plt.subplot(1, 2, 2)
    plt.title('Computed undistortion')

    w, h = img_size

    x_max = int(w / 2) #  change back to 2 instead of 4
    y_max = int(h / 2)

    for x_idx, x_d in enumerate(x_range):
        x_idx = x_idx - x_max
        for y_idx, y_d in enumerate(y_range):
            y_idx = y_idx - y_max

            x_u, y_u = undist_dict[(x_idx, y_idx)]

            dx = x_u - x_d
            dy = y_u - y_d

            plt.plot(x_d, y_d, 'bo', markersize=4)
            plt.plot(x_u, y_u, 'ro', markersize=4)
            plt.quiver(x_d, y_d, dx, dy, angles='xy', scale_units='xy', scale=1, width=0.005, headwidth=3)

    plt.legend(("Distorted point", "Undistorted point"))
    plt.show()


def radial_distortion_model(undistorted_coord, distorted_coord, k1):
    # These coordinates have their origin at the camera principal point
    x_d = distorted_coord[0]
    y_d = distorted_coord[1]
    x_u = undistorted_coord[0]
    y_u = undistorted_coord[1]

    f1 = x_u * (1 + k1 * (x_u ** 2 + y_u ** 2)) - x_d
    f2 = y_u * (1 + k1 * (x_u ** 2 + y_u ** 2)) - y_d

    return np.array([f1, f2])


def compute_radial_undistortion_mapping(k, img_size):
    """
    :param k: float, radial distortion parameter
    :param img_size: tuple (w,h), size of distorted image
    :return: dict[(x_dist,y_dist)] = np.array([x_undist, y_undist]).
             Note: the coordinates has origin at the principal point.
    """
    undistortion_dict = {}
    w, h = img_size

    # +- principal point max
    w_max = int(np.ceil(w / 2))
    h_max = int(np.ceil(h / 2))

    w_range = np.linspace(-w_max, w_max, w)
    h_range = np.linspace(-h_max, h_max, h)

    for w_idx, w_i in enumerate(w_range):
        w_idx = w_idx - w_max
        for h_idx, h_i in enumerate(h_range):
            h_idx = h_idx - h_max

            distorted_coord = np.array([w_i / (w/2), h_i / (h/2)])

            func = lambda undistorted_coord: radial_distortion_model(undistorted_coord, distorted_coord, k)
            undistortion_dict[(w_idx, h_idx)] = fsolve(func, distorted_coord)

            print("for the distorted coord: ", distorted_coord, "\nfsolve computes the undistorted coord: ",  undistortion_dict[(w_idx, h_idx)], "\nyielding an error of: ", func((undistortion_dict[(w_idx, h_idx)])), "\n")

    return undistortion_dict


k = -0.2
# Keep at 10,10 for now otherwise things may break
img_size = (10, 10)

# try -0.5 to 0.5 (prior to multiplying with k^-1)

visualize_radial_model(k)

undist_dict = compute_radial_undistortion_mapping(k, img_size)

visualize_distortion(undist_dict, k, img_size)

