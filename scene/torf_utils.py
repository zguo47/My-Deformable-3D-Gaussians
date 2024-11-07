import os
import cv2
import scipy
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import io


# Image utils
def to8b(x):
    return (255*np.clip(x, 0, 1)).astype(np.uint8)

def normalize_im_max(im):
    im = im / np.max(im)
    im[np.isnan(im)] = 0.
    return im

def normalize_im(im):
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    im[np.isnan(im)] = 0.
    return np.clip(im, 0, 1)

def normalize_im_gt(im, im_gt):
    im = (im - np.min(im_gt)) / (np.max(im_gt) - np.min(im_gt))
    im[np.isnan(im)] = 0.
    return np.clip(im, 0, 1)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def resize_all_images(images, width, height, method=cv2.INTER_AREA):
    resized_images = []
    for i in range(images.shape[0]):
        resized_images.append(cv2.resize(images[i], (width, height), interpolation=method))
    return np.stack(resized_images, axis=0)

def scale_image(image, scale=1, interpolation=cv2.INTER_AREA):
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)


# ToF/Depth utils
def depth_from_tof(tof, depth_range, phase_offset=0.0):
    tof_phase = np.arctan2(tof[..., 1:2], tof[..., 0:1])
    tof_phase -= phase_offset
    tof_phase[tof_phase < 0] = tof_phase[tof_phase < 0] + 2 * np.pi
    return tof_phase * depth_range / (4 * np.pi)

def tof_from_depth(depth, amp, depth_range):
    tof_phase = depth * 4 * np.pi / depth_range
    amp *= 1. / np.maximum(depth * depth, (depth_range * 0.1) * (depth_range * 0.1))
    return np.stack([np.cos(tof_phase) * amp, np.sin(tof_phase) * amp, amp], axis=-1)

def z_depth_to_distance_map(z_depth, K):
    x, y = np.meshgrid(np.arange(z_depth.shape[1]), np.arange(z_depth.shape[0]))
    return np.sqrt(((x - K[0, 2]) * z_depth / K[0, 0]) ** 2 + ((y - K[1, 2]) * z_depth / K[1, 1]) ** 2 + z_depth ** 2)

def distance_to_z_depth(distance_map, K):
    x, y = np.meshgrid(np.arange(distance_map.shape[1]), np.arange(distance_map.shape[0]))
    return distance_map / np.sqrt(((x - K[0, 2]) / K[0, 0]) ** 2 + ((y - K[1, 2]) / K[1, 1]) ** 2 + 1)


# Camera utils
def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       raise ValueError("Cannot normalize a zero vector")
    return v / norm

def get_camera_params(intrinsics_file, extrinsics_file, total_num_views):
    if '.mat' in intrinsics_file:
        K = scipy.io.loadmat(intrinsics_file)['K']
    else:
        K = np.load(intrinsics_file)
    Ks = [np.copy(K) for _ in range(total_num_views)]

    exts = np.load(extrinsics_file)
    return Ks, exts

def se3_vee(mat):
    mat = tf.linalg.logm(tf.cast(mat, tf.complex64))
    twist = tf.stack(
        [
            mat[..., 2, 1],
            mat[..., 0, 2],
            mat[..., 1, 0],
            mat[..., 0, 3],
            mat[..., 1, 3],
            mat[..., 2, 3],
        ],
        axis=-1
        )
    
    return tf.cast(twist, tf.float32)

def se3_hat(twist):
    twist = tf.cast(twist, tf.complex64)
    null = tf.zeros_like(twist[..., 0])

    mat = tf.stack(
        [
            tf.stack(
                [
                null,
                twist[..., 2],
                -twist[..., 1],
                null
                ],
                axis=-1
            ),
            tf.stack(
                [
                -twist[..., 2],
                null,
                twist[..., 0],
                null
                ],
                axis=-1
            ),
            tf.stack(
                [
                twist[..., 1],
                -twist[..., 0],
                null,
                null
                ],
                axis=-1
            ),
            tf.stack(
                [
                twist[..., 3],
                twist[..., 4],
                twist[..., 5],
                null
                ],
                axis=-1
            ),
        ],
        axis=-1
        )
    
    return tf.cast(tf.linalg.expm(mat), tf.float32)

def normalize(v):
    return tf.math.l2_normalize(v, axis=-1, epsilon=1e-6)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses): # c2w
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    hwf = c2w[:,4:5]
    rads = np.array(list(rads) + [1.])
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.sin(-theta), np.cos(-theta), np.sin(-theta * zrate), 1.]) * rads)
        z = normalize(-c + np.dot(c2w[:3,:4], np.array([0, 0, focal, 1.])))
        pose = np.eye(4)
        pose[:3, :4] = viewmatrix(z, up, c)
        render_poses.append(pose)

    return render_poses

def get_render_poses_spiral(focal_length, bounds_data, intrinsics, poses, scene_scale, N_views=60, N_rots=2):
    intrinsics = np.array(intrinsics)
    poses = np.array(poses)

    ## Focus distance
    if focal_length < 0:
        close_depth, inf_depth = bounds_data.min() * .9, bounds_data.max() * 5.
        dt = .75
        mean_dz = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)
        focal_length = mean_dz

    # Get average pose
    c2w = poses_avg(poses)
    c2w_path = c2w
    up = normalize(poses[:, :3, 1].sum(0))

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = bounds_data.min() * .2
    tt = (poses[:, :3, 3] - c2w[:3, 3])
    if np.sum(tt) < 1e-10:
        tt = np.array([1.0, 1.0, 1.0])

    rads = np.percentile(np.abs(tt), 90, 0) * np.array([1.0, 1.0, 1.0]) * scene_scale / 16.0
    # light_rads = np.percentile(np.abs(tt), 90, 0) * np.array([1.0, 1.0, 1.0])

    # Generate poses for spiral path
    render_poses = render_path_spiral(c2w_path, up, rads, focal_length, zdelta, zrate=.5, rots=N_rots, N=N_views)
    render_poses = np.array(render_poses).astype(np.float32)

    # render_light_poses = render_path_spiral(c2w_path, up, light_rads, focal_length, zdelta, zrate=.5, rots=N_rots, N=N_views)
    # render_light_poses = np.array(render_light_poses).astype(np.float32)

    # return render_poses, render_light_poses
    return render_poses

def cameraFrustumCorners(cam_info):
    """
    Calculate the world-space positions of the corners of the camera's view frustum.
    """
    aspect_ratio = cam_info.width / cam_info.height
    hnear = 2 * np.tan(cam_info.FovY / 2) * cam_info.znear
    wnear = hnear * aspect_ratio
    hfar = 2 * np.tan(cam_info.FovX / 2) * cam_info.zfar
    wfar = hfar * aspect_ratio

    # Camera's forward direction (z forward y down in SfM convention)
    forward = normalize_vector(np.linalg.inv(np.transpose(cam_info.R))[:, 2])
    right = normalize_vector(np.linalg.inv(np.transpose(cam_info.R))[:, 0])
    up = -normalize_vector(np.linalg.inv(np.transpose(cam_info.R))[:, 1])

    # Camera position
    cam_pos = -np.linalg.inv(np.transpose(cam_info.R)) @ cam_info.T

    # Near plane corners
    nc_tl = cam_pos + forward * cam_info.znear + up * (hnear / 2) - right * (wnear / 2)
    nc_tr = cam_pos + forward * cam_info.znear + up * (hnear / 2) + right * (wnear / 2)
    nc_bl = cam_pos + forward * cam_info.znear - up * (hnear / 2) - right * (wnear / 2)
    nc_br = cam_pos + forward * cam_info.znear - up * (hnear / 2) + right * (wnear / 2)

    # Far plane corners
    fc_tl = cam_pos + forward * cam_info.zfar + up * (hfar / 2) - right * (wfar / 2)
    fc_tr = cam_pos + forward * cam_info.zfar + up * (hfar / 2) + right * (wfar / 2)
    fc_bl = cam_pos + forward * cam_info.zfar - up * (hfar / 2) - right * (wfar / 2)
    fc_br = cam_pos + forward * cam_info.zfar - up * (hfar / 2) + right * (wfar / 2)

    return np.array([nc_tl, nc_tr, nc_bl, nc_br, fc_tl, fc_tr, fc_bl, fc_br])

def calculateSceneBounds(cam_infos, args):
    cam_xyzs = np.array([-np.linalg.inv(np.transpose(cam_info.R)) @ cam_info.T for cam_info in cam_infos])
    cam_dirs = np.array([normalize_vector(np.linalg.inv(np.transpose(cam_info.R))[:, 2]) for cam_info in cam_infos]) # SfM convention
    
    all_corners = []
    for cam_info in cam_infos:
        corners = cameraFrustumCorners(cam_info)
        all_corners.append(corners)
    
    # if args.debug:
    plt.ioff()
    # Visualize camera positions
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(cam_xyzs[:, 0], cam_xyzs[:, 1], cam_xyzs[:, 2], color = "green")

    # Visualize camera viewing directions
    for i in range(cam_dirs.shape[0]):
        view_dir = cam_dirs[i]
        scale = 0.05
        ax.quiver(cam_xyzs[i, 0], cam_xyzs[i, 1], cam_xyzs[i, 2], view_dir[0]*scale, view_dir[1]*scale, view_dir[2]*scale, color='red', length=3, normalize=True)

    # Visualize camera corners (to determine scene bounds)
    for cs in all_corners:
        ax.scatter3D(cs[:, 0], cs[:, 1], cs[:, 2], color = "blue")
    plt.title("Camera Poses")
    plt.legend()
    plt.savefig(os.path.join(args.model_path, "scene_bounds.png"))
    # plt.show()
    plt.close()

    all_corners = np.vstack(all_corners)
    min_bounds = np.min(all_corners, axis=0)
    max_bounds = np.max(all_corners, axis=0)

    return min_bounds, max_bounds