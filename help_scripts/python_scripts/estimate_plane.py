import numpy as np
import math


def ransac_find_plane(pts, threshold):
# pts: Nx3, N 3D points
# threshold: Scalar, threshold , threshold for points to be inliers
# plane: 4x1, plane on the form ax + by + cz + d = 0

    print("\nEstimating floor plane...")

    if threshold == 0:
        print('Threshold = 0 may give false outliers due to machine precision errors')

    # Init
    N = len(pts)
    epsilon = 0.1  # epsilon0

    mismatch_prob = 0.1  # eta

    kmax = int(math.log(mismatch_prob) / math.log(1 - math.pow(epsilon, 3)))
    min_outliers = N
    k = 1

    while k < kmax:
        # Select subset of points and calculate preliminary plane
        subset = np.random.permutation(N)  # randomize 3 points
        pts_prim = pts[subset[0:3], :]

        plane_prel = compute_plane(pts_prim)

        # Measure performance of prel plane
        residual_lengths = residual_lengths_points_to_plane(pts, plane_prel)

        outliers = sum(residual_lengths > threshold)
        inliers = N - outliers

        # Log keeping
        if (outliers > 0):
            if outliers < min_outliers:
                # if best sub-perfect case found, save loss and iterate
                min_outliers = outliers
                plane = plane_prel
                # Commenting these out as we're not really in a hurry -> take time to find best plane
                #epsilon = inliers / N
                #kmax = math.log(mismatch_prob) / math.log(1 - math.pow(epsilon, 3))
        else:
            # If best case found(outliers=0), return immediately
            print('# of outliers !> 0. (THIS CASE HAS NOT YET BEEN TESTED)\nIF THIS SHOWS; SOMETHING IS LIKELY WRONG')
            plane = plane_prel
            min_outliers = outliers
            print('Total # of iterations was ' + str(k) + ' with 0% outliers.')
            return plane, min_outliers

        print("", end="\r")
        print("{:.2f} % done.".format(100 * (k + 1) / kmax), end="")
        k = k + 1

    print('\nTotal # of RANSAC iterations was {} and optimal percentage of outliers was {:.2f} %\n'.format(k, (100*min_outliers/N)))
    return plane, min_outliers

def compute_plane(pts):
    """
    :param pts: 3x3, 3 3D points of form 3x1
    :return: 4x1, plane such that Ax + By + Cz + D = 0
    """

    A = pts[0, :]
    B = pts[1, :]
    C = pts[2, :]

    AB = B - A
    AC = C - A

    N = np.cross(AB, AC)

    plane = [N[0], N[1], N[2], -(N[0]*A[0] + N[1]*A[1] + N[2]*A[2])]  # sum(np.multiply(-C, N))]
    return plane

def residual_lengths_points_to_plane(pts, plane):
    """
    :param pts: pts: 3xN 3D points
    :param plane: 4x1 [a,b,c,d] such that ax+by+cz+d=0
    :return residual_lengths: 1xN the minimum distance from all points to the plane
    """

    N = len(pts[:, 0])
    normal_vec = np.divide(plane[:3], math.sqrt(sum(np.power(plane[:3], 2))))
    residual_lengths = np.zeros(N)

    # find a point on the plane
    if not math.isclose(plane[0], 0, rel_tol=1e-04, abs_tol=1e-05):
        P = [-plane[3]/plane[0], 0, 0]
    elif not math.isclose(plane[1], 0, rel_tol=1e-04, abs_tol=1e-05):
        P = [0, -plane[3] / plane[1], 0]
    elif not math.isclose(plane[2], 0, rel_tol=1e-04, abs_tol=1e-05):
        P = [0, 0, -plane[3] / plane[2]]
    else:
        P = [0, 0, 0]

    for i in range(0, N):
        #difference vector from plane to point
        u = pts[i, :] - P

        #length of difference vector projected onto normal vector
        residual_lengths[i] = abs(np.dot(u, normal_vec))

    return residual_lengths

