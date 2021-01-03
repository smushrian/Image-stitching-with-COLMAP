from help_scripts.python_scripts.COLMAP_functions import *
from help_scripts.python_scripts.estimate_plane import ransac_find_plane
from help_scripts.python_scripts.color_virtual_image import *
from help_scripts.python_scripts.undistortion import compute_all_maps
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
from scipy.spatial.transform import Rotation as R
from sympy import Matrix
import time


def plot_3D(xyz, plane, all_cameras, cam_virt):

    plt3d = plt.figure("Visualization of floor plane").gca(projection='3d', autoscale_on=False)
    colors = {1: 'r', 2: 'b', 3: 'g', 4: 'c'}

    a, b, c, d = plane

    x = np.linspace(-5, 7, 10)
    y = np.linspace(-5, 7, 10)
    X, Y = np.meshgrid(x, y)
    Z = (d + a * X + b * Y) / -c

    plt3d.plot_surface(X, Y, Z, alpha=0.5, label="Floor plane")  # plot plane
    plt3d.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2],  cmap='Greens', label="COLMAP 3D points")  # plot COLMAP 3D points

    # Plot virtual camera position
    cam_center_virt, principal_axis_virt = get_camera_center_and_axis(cam_virt)
    plt3d.quiver(cam_center_virt[0, 0], cam_center_virt[1, 0], cam_center_virt[2, 0], principal_axis_virt[0, 0],
                 principal_axis_virt[0, 1], principal_axis_virt[0, 2], length=2, color='r', label="Virtual Camera")

    # Plot all physical camera positions
    for key in all_cameras.keys():
        cam_center, principal_axis = get_camera_center_and_axis(all_cameras[key]['P'])
        plt3d.quiver(cam_center[0,0],cam_center[1,0],cam_center[2,0], principal_axis[0,0],
                     principal_axis[0,1], principal_axis[0,2], length=2, color=colors[2], label="Physical cameras")

    # plt3d.legend()  # It appears theres bug preventing us to use legend here
    plt3d.set_xlim3d(-5, 7)
    plt3d.set_ylim3d(-5, 7)
    plt3d.set_zlim3d(0, 12)
    plt3d.view_init(elev=-175, azim=-83)


def load_data_from_colmap(data_dir=None, img_dir=None):

    if data_dir == None:
        data_dir = os.getcwd() + "/COLMAP_w_CUDA"
    if img_dir == None:
        img_dir = os.getcwd() + "/COLMAP_w_CUDA/images/"

    cameras, points3D, images = get_data_from_binary(data_dir)

    # Load all 3D point coordinates from COLMAP
    pts3Dcoords = []
    for key in points3D:
        pts3Dcoords.append(points3D[key].xyz)
    pts3Dcoords = np.asarray(pts3Dcoords)

    # Load all camera matrices and images
    camera_intrinsics = {}
    all_camera_matrices = {}
    imgs = {}

    # Rearrange COLMAP data
    # everything is saved in dicts with the corresponding camera ID as key
    for key in images.keys():

        imgs[images[key].camera_id] = plt.imread(img_dir + images[key].name)
        all_camera_matrices[images[key].camera_id] = camera_quat_to_P(images[key].qvec, images[key].tvec)
        camera_intrinsics[cameras[key].id] = cameras[key]

    return pts3Dcoords, imgs, all_camera_matrices, camera_intrinsics


def generate_camera_params(all_camera_matrices, camera_intrinsics, plane, cam_type="virtual"):
    if cam_type == "virtual":
        # VIRTUAL CAMERA WITH MEAN CENTER OF OTHER CAMERAS AND PRINCIPAL AXIS AS PLANE NORMAL
        Pv = create_virtual_camera(all_camera_matrices, plane)
        w = 500
        h = 500
        f = 250
        K_virt = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])

    elif cam_type == "existing":
        # EXISTING CAMERA
        K_virt, _ = build_intrinsic_matrix(camera_intrinsics[1])
        Pv = all_camera_matrices[1]['P']
        f, cx, cy, _ = camera_intrinsics[1].params
        w, h = int(2*cx), int(2*cy)

    else:
        print("Undefined cam type")
        raise ValueError

    return Pv, w, h, f, K_virt


def get_camera_center_and_axis(P):
    P = Matrix(P)
    cam_center = P.nullspace()[0]
    principal_axis = P[2, :3]
    return np.asarray(cam_center), np.asarray(principal_axis)


def camera_quat_to_P(quat, t):
    quat_scalar_last = [quat[1],quat[2],quat[3],quat[0]]
    R_matrix = R.from_quat(quat_scalar_last).as_matrix()
    t = np.asarray(t)
    P = np.column_stack((R_matrix,t))
    cam = {'P': P, 'R': R_matrix, 't': t}
    return cam


def create_virtual_camera(camera_matrices,plane):
    centers = {}
    axes = {}
    for index,cam in enumerate(camera_matrices):
        cam_center, principal_axis = get_camera_center_and_axis(camera_matrices[cam]['P'])
        centers[index] = cam_center
        axes[index] = principal_axis
    virt_center = (centers[0]+centers[1]+centers[2]+centers[3])/4

    plane = plane/plane[3]
    virt_principal_axis = np.asarray([plane[0],plane[1],plane[2]],dtype='float')/np.linalg.norm([plane[0],plane[1],plane[2]])#(axes[0]+axes[1]+axes[2]+axes[3])/4

    virt_principal_axis = -virt_principal_axis
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])

    xnew = np.cross(x,virt_principal_axis)
    normx = math.sqrt(math.pow(xnew[0],2) + math.pow(xnew[1],2) + math.pow(xnew[2],2))
    xnew = xnew/normx

    ynew = np.cross(y,virt_principal_axis)
    normy = math.sqrt(math.pow(ynew[0],2) + math.pow(ynew[1],2) + math.pow(ynew[2],2))
    ynew = ynew/normy

    Rvirt = np.asarray([[xnew[0], xnew[1], xnew[2]],[ynew[0], ynew[1], ynew[2]],
                        [virt_principal_axis[0],virt_principal_axis[1],virt_principal_axis[2]]],dtype='float')

    tvirt = np.asarray(np.matmul(-Rvirt,np.asarray([virt_center[0][0],virt_center[1][0],virt_center[2][0]])),dtype='float')

    Pvirt = np.column_stack((Rvirt,tvirt))

    return Pvirt


def generate_homography_matrices(all_camera_matrices, camera_intrinsics, Pv, K_virt, plane):
    H = {}
    P_real_new = {}
    for key in all_camera_matrices.keys():
        K_temp, _ = build_intrinsic_matrix(camera_intrinsics[key])
        H[key], plane_new, P_real_new[key], P_virt_trans = compute_homography(Pv, all_camera_matrices[key]['P'], K_virt,
                                                                              K_temp, plane)
    return H, plane_new, P_real_new, P_virt_trans


# Perform the reconstruction to get data
# automatic_reconstructor()
# image_undistorter()
# stereo_fusion()


pts3Dcoords, imgs, all_camera_matrices, camera_intrinsics = load_data_from_colmap()

# Estimate floor plane
plane, _ = ransac_find_plane(pts3Dcoords, threshold=0.01)

# Set stitching viewpoint parameters
Pv, w, h, f, K_virt = generate_camera_params(all_camera_matrices, camera_intrinsics, plane, cam_type="virtual")

# Visualize floor plane, 3D points, and all cameras
plot_3D(pts3Dcoords, plane, all_camera_matrices, Pv)

# Construct undistortion maps
maps = compute_all_maps(r'/Users/ludvig/Documents/SSY226 Design project in MPSYS/Image-stitching-with-COLMAP/COLMAP_w_CUDA/', full_size_img=False)

# Construct homography matrices
H, plane_new, P_real_new, P_virt_trans = generate_homography_matrices(all_camera_matrices, camera_intrinsics, Pv, K_virt, plane)

start = time.process_time()
for img_key in imgs.keys():
    map_x, map_y = maps[img_key]
    imgs[img_key] = cv.remap(imgs[img_key], map_x, map_y, cv.INTER_LANCZOS4)

# Stitch image
color_images, stitched_image = stitch_image(plane, Pv, w, h, imgs, all_camera_matrices, camera_intrinsics,
                                            K_virt, 'homography', H)
end = time.process_time()

print("\nImage undistortion + stitching took {:.2f} seconds.".format(end-start))

plt.figure("Final image")
plt.imshow(stitched_image)

# PLOT TRANSFORMED PLANE WITH TRANSFORMED CAMERAS (HOMOGRAPHY)
a, b, c, d = plane_new
x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)
X, Y = np.meshgrid(x, y)
Z = (d + a * X + b * Y) / -c
plt4d = plt.figure().gca(projection='3d', autoscale_on=False)
plt4d.plot_surface(X, Y, Z, alpha=0.5)
cam_center, principal_axis = get_camera_center_and_axis(P_virt_trans)
plt4d.quiver(cam_center[0],cam_center[1],cam_center[2], principal_axis[0,0], principal_axis[0,1], principal_axis[0,2], length=d, color='r')
colors = {1: 'r', 2: 'b', 3: 'g', 4: 'c'}
for key in P_real_new:
    cam_center, principal_axis = get_camera_center_and_axis(P_real_new[key])
    plt4d.quiver(cam_center[0],cam_center[1],cam_center[2], principal_axis[0,0], principal_axis[0,1], principal_axis[0,2], length=1, color=colors[key])


# Test for visualizing the projection of a virtual pixel to the plane
# pixelpoint = [0,0]
# line_dir,line_point = line_from_pixel(pixelpoint,Pv,K_virt)
# intersection_point = intersection_line_plane(line_dir,line_point,plane)
# plt3d.scatter3D(intersection_point[0], intersection_point[1], intersection_point[2],  cmap='Blues')
# plt3d.quiver(line_point[0],line_point[1],line_point[2], line_dir[0], line_dir[1], line_dir[2], length=10, color='y')
# cam_center, principal_axis = get_camera_center_and_axis(Pv)
# plt3d.quiver(cam_center[0],cam_center[1],cam_center[2], principal_axis[0,0], principal_axis[0,1], principal_axis[0,2], length=distance, color='Red')
plt.show()

