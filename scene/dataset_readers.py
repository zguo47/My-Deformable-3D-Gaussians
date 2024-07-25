#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from scene.torf_utils import get_camera_params, normalize_im_max, scale_image, calculateSceneBounds
import numpy as np
import json
import imageio
from glob import glob
import cv2 as cv
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB, RGB2SH, SH2PA, PA2SH
from scene.gaussian_model import BasicPointCloud
from utils.camera_utils import camera_nerfies_from_JSON
from tqdm import tqdm


class CameraInfo(NamedTuple):
    # uid: int
    # R: np.array
    # T: np.array
    # FovY: np.array
    # FovX: np.array
    # image: np.array
    # image_path: str
    # image_name: str
    # width: int
    # height: int
    fid: float
    uid: int
    # Color
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image_name: str
    image: np.array
    image_path: str
    width: int
    height: int
    frame_id: Optional[int] = -1
    # Time-of-Flight
    R_tof: Optional[np.array] = None
    T_tof: Optional[np.array] = None
    FovY_tof: Optional[np.array] = None
    FovX_tof: Optional[np.array] = None
    tof_image_name: Optional[str] = ""
    tof_image: Optional[np.array] = None
    tof_image_path: Optional[str] = ""
    distance_image_name: Optional[str] = ""     # Distance image (for synthetic scenes)
    distance_image: Optional[np.array] = None
    distance_image_path: Optional[str] = ""
    tof_width: Optional[int] = -1
    tof_height: Optional[int] = -1
    # Others
    znear: Optional[float] = 0.01
    zfar: Optional[float] = 100.0
    depth_range: Optional[float] = 100.0
    depth: Optional[np.array] = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]]
                 for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return K, pose


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    num_frames = len(cam_extrinsics)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(
            "Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        fid = int(image_name) / (num_frames - 1)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, fid=fid)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'],
                       vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            frame_time = frame['time']

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array(
                [1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            mask = norm_data[..., 3:4]

            arr = norm_data[:, :, :3] * norm_data[:, :,
                                                  3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(
                np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovx
            FovX = fovy

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[
                                            0],
                                        height=image.size[1], fid=frame_time))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readDTUCameras(path, render_camera, object_camera):
    camera_dict = np.load(os.path.join(path, render_camera))
    images_lis = sorted(glob(os.path.join(path, 'image/*.png')))
    masks_lis = sorted(glob(os.path.join(path, 'mask/*.png')))
    n_images = len(images_lis)
    cam_infos = []
    cam_idx = 0
    for idx in range(0, n_images):
        image_path = images_lis[idx]
        image = np.array(Image.open(image_path))
        mask = np.array(imageio.imread(masks_lis[idx])) / 255.0
        image = Image.fromarray((image * mask).astype(np.uint8))
        world_mat = camera_dict['world_mat_%d' % idx].astype(np.float32)
        fid = camera_dict['fid_%d' % idx] / (n_images / 12 - 1)
        image_name = Path(image_path).stem
        scale_mat = camera_dict['scale_mat_%d' % idx].astype(np.float32)
        P = world_mat @ scale_mat
        P = P[:3, :4]

        K, pose = load_K_Rt_from_P(None, P)
        a = pose[0:1, :]
        b = pose[1:2, :]
        c = pose[2:3, :]

        pose = np.concatenate([a, -c, -b, pose[3:, :]], 0)

        S = np.eye(3)
        S[1, 1] = -1
        S[2, 2] = -1
        pose[1, 3] = -pose[1, 3]
        pose[2, 3] = -pose[2, 3]
        pose[:3, :3] = S @ pose[:3, :3] @ S

        a = pose[0:1, :]
        b = pose[1:2, :]
        c = pose[2:3, :]

        pose = np.concatenate([a, c, b, pose[3:, :]], 0)

        pose[:, 3] *= 0.5

        matrix = np.linalg.inv(pose)
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        FovY = focal2fov(K[0, 0], image.size[1])
        FovX = focal2fov(K[0, 0], image.size[0])
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image.size[
                                  0], height=image.size[1],
                              fid=fid)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readNeuSDTUInfo(path, render_camera, object_camera):
    print("Reading DTU Info")
    train_cam_infos = readDTUCameras(path, render_camera, object_camera)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=[],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readNerfiesCameras(path):
    with open(f'{path}/scene.json', 'r') as f:
        scene_json = json.load(f)
    with open(f'{path}/metadata.json', 'r') as f:
        meta_json = json.load(f)
    with open(f'{path}/dataset.json', 'r') as f:
        dataset_json = json.load(f)

    coord_scale = scene_json['scale']
    scene_center = scene_json['center']

    name = path.split('/')[-2]
    if name.startswith('vrig'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 0.25
    elif name.startswith('NeRF'):
        train_img = dataset_json['train_ids']
        val_img = dataset_json['val_ids']
        all_img = train_img + val_img
        ratio = 1.0
    elif name.startswith('interp'):
        all_id = dataset_json['ids']
        train_img = all_id[::4]
        val_img = all_id[2::4]
        all_img = train_img + val_img
        ratio = 0.5
    else:  # for hypernerf
        train_img = dataset_json['ids'][::4]
        all_img = train_img
        ratio = 0.5

    train_num = len(train_img)

    all_cam = [meta_json[i]['camera_id'] for i in all_img]
    all_time = [meta_json[i]['time_id'] for i in all_img]
    max_time = max(all_time)
    all_time = [meta_json[i]['time_id'] / max_time for i in all_img]
    selected_time = set(all_time)

    # all poses
    all_cam_params = []
    for im in all_img:
        camera = camera_nerfies_from_JSON(f'{path}/camera/{im}.json', ratio)
        camera['position'] = camera['position'] - scene_center
        camera['position'] = camera['position'] * coord_scale
        all_cam_params.append(camera)

    all_img = [f'{path}/rgb/{int(1 / ratio)}x/{i}.png' for i in all_img]

    cam_infos = []
    for idx in range(len(all_img)):
        image_path = all_img[idx]
        image = np.array(Image.open(image_path))
        image = Image.fromarray((image).astype(np.uint8))
        image_name = Path(image_path).stem

        orientation = all_cam_params[idx]['orientation'].T
        position = -all_cam_params[idx]['position'] @ orientation
        focal = all_cam_params[idx]['focal_length']
        fid = all_time[idx]
        T = position
        R = orientation

        FovY = focal2fov(focal, image.size[1])
        FovX = focal2fov(focal, image.size[0])
        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image.size[
                                  0], height=image.size[1],
                              fid=fid)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos, train_num, scene_center, coord_scale


def readNerfiesInfo(path, eval):
    print("Reading Nerfies Info")
    cam_infos, train_num, scene_center, scene_scale = readNerfiesCameras(path)

    if eval:
        train_cam_infos = cam_infos[:train_num]
        test_cam_infos = cam_infos[train_num:]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        print(f"Generating point cloud from nerfies...")

        xyz = np.load(os.path.join(path, "points.npy"))
        xyz = (xyz - scene_center) * scene_scale
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromNpy(path, npy_file, split, hold_id, num_images):
    cam_infos = []
    video_paths = sorted(glob(os.path.join(path, 'frames/*')))
    poses_bounds = np.load(os.path.join(path, npy_file))

    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    H, W, focal = poses[0, :, -1]

    n_cameras = poses.shape[0]
    poses = np.concatenate(
        [poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    bottoms = np.array([0, 0, 0, 1]).reshape(
        1, -1, 4).repeat(poses.shape[0], axis=0)
    poses = np.concatenate([poses, bottoms], axis=1)
    poses = poses @ np.diag([1, -1, -1, 1])

    i_test = np.array(hold_id)
    video_list = i_test if split != 'train' else list(
        set(np.arange(n_cameras)) - set(i_test))

    for i in video_list:
        video_path = video_paths[i]
        c2w = poses[i]
        images_names = sorted(os.listdir(video_path))
        n_frames = num_images

        matrix = np.linalg.inv(np.array(c2w))
        R = np.transpose(matrix[:3, :3])
        T = matrix[:3, 3]

        for idx, image_name in enumerate(images_names[:num_images]):
            image_path = os.path.join(video_path, image_name)
            image = Image.open(image_path)
            frame_time = idx / (n_frames - 1)

            FovX = focal2fov(focal, image.size[0])
            FovY = focal2fov(focal, image.size[1])

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovX=FovX, FovY=FovY,
                                        image=image,
                                        image_path=image_path, image_name=image_name,
                                        width=image.size[0], height=image.size[1], fid=frame_time))

            idx += 1
    return cam_infos


def readPlenopticVideoDataset(path, eval, num_images, hold_id=[0]):
    print("Reading Training Camera")
    train_cam_infos = readCamerasFromNpy(path, 'poses_bounds.npy', split="train", hold_id=hold_id,
                                         num_images=num_images)

    print("Reading Training Camera")
    test_cam_infos = readCamerasFromNpy(
        path, 'poses_bounds.npy', split="test", hold_id=hold_id, num_images=num_images)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, 'points3D.ply')
    if not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


# ToRF scenes
def readToRFCameras(path, tof_extrinsics, tof_intrinsics, color_extrinsics, color_intrinsics, depth_range, znear, zfar, args):
    cam_infos = []

    for fid in tqdm(range(args.total_num_views), desc="Loading all views/frames"):
        # Color camera
        R = np.transpose(color_extrinsics[fid, :3, :3]) # torf extrinsics is w2c
        T = color_extrinsics[fid, :3, 3]
        FovY = 2 * np.arctan2(args.color_image_height, 2 * color_intrinsics[fid][1, 1]) # radian
        FovX = 2 * np.arctan2(args.color_image_width, 2 * color_intrinsics[fid][0, 0])
        
        color_image_name = f"{fid:04d}"
        color_image_path = os.path.join(path, "color", f"{color_image_name}.npy")
        color_image = Image.fromarray(np.array(normalize_im_max(scale_image(np.load(color_image_path), args.color_scale_factor)) * 255.0, dtype=np.byte), "RGB")

        # Time-of-Flight camera
        R_tof = np.transpose(tof_extrinsics[fid, :3, :3])
        T_tof = tof_extrinsics[fid, :3, 3]
        FovY_tof = 2 * np.arctan2(args.tof_image_height, 2 * tof_intrinsics[fid][1, 1])
        FovX_tof = 2 * np.arctan2(args.tof_image_width, 2 * tof_intrinsics[fid][0, 0])

        tof_image_name = f"{fid:04d}"
        tof_image_path = os.path.join(path, "tof", f"{tof_image_name}.npy")
        tof_image = normalize_im_max(scale_image((np.load(tof_image_path)), args.tof_scale_factor))

        # Distance image (for synthetic scenes)        
        distance_image_name = f"{fid:04d}" 
        distance_image_path = os.path.join(path, "distance", f"{distance_image_name}.npy")
        distance_image = np.load(distance_image_path)

        cam_infos.append(CameraInfo(
            uid=fid, fid=fid, frame_id=fid, 
            # Color
            R=R, T=T, FovY=FovY, FovX=FovX, 
            image_name=color_image_name, image=color_image, image_path=color_image_path,
            width=args.color_image_width*args.color_scale_factor, height=args.color_image_height*args.color_scale_factor,
            # Time-of-Flight
            R_tof=R_tof, T_tof=T_tof, FovY_tof=FovY_tof, FovX_tof=FovX_tof, 
            tof_image_name=tof_image_name, tof_image=tof_image, tof_image_path=tof_image_path,
            distance_image_name=distance_image_name, distance_image=distance_image, distance_image_path=distance_image_path,
            tof_width=args.tof_image_width*args.tof_scale_factor, tof_height=args.tof_image_height*args.tof_scale_factor,
            # Others
            znear=znear, zfar=zfar, depth_range=depth_range))
    return cam_infos

def readToRFSpiralCameras(extrinsics, intrinsics, depth_range, znear, zfar, args):
    cam_infos = []
    
    for fid in tqdm(range(args.total_num_spiral_views)):
        R = np.transpose(extrinsics[fid, :3, :3])
        T = extrinsics[fid, :3, 3]
        FovY = 2 * np.arctan2(args.tof_image_height, 2 * intrinsics[fid][1, 1])
        FovX = 2 * np.arctan2(args.tof_image_width, 2 * intrinsics[fid][0, 0])
        
        cam_infos.append(CameraInfo(uid=fid, frame_id=fid,
                                    R=R, T=T, FovY=FovY, FovX=FovX, 
                                    image_name=f"{fid:04d}", image=None, image_path=None,
                                    width=args.color_image_width*args.color_scale_factor, height=args.color_image_height*args.color_scale_factor, 
                                    znear=znear, zfar=zfar, depth_range=depth_range))
    return cam_infos

def readToRFSceneInfo(path, eval, args, llffhold=8):
    # Load cameras
    if args.dataset_type == "real":
        cam_file_ending = 'mat'
    else:
        cam_file_ending = 'npy'

    tof_intrinsics, tof_extrinsics = get_camera_params(
        os.path.join(path, 'cams', f'tof_intrinsics.{cam_file_ending}'), 
        os.path.join(path, 'cams', 'tof_extrinsics.npy'), 
        args.total_num_views)
    color_intrinsics, color_extrinsics = get_camera_params(
        os.path.join(path, 'cams', f'color_intrinsics.{cam_file_ending}'), 
        os.path.join(path, 'cams', 'color_extrinsics.npy'), 
        args.total_num_views)
    
    depth_range_path = os.path.join(path, 'cams', 'depth_range.npy')
    if os.path.exists(depth_range_path):
        depth_range = np.load(depth_range_path).astype(np.float32)
    else:
        depth_range = np.array(args.depth_range).astype(np.float32)
    znear = args.min_depth_fac * depth_range
    zfar = args.max_depth_fac * depth_range
    
    # Create splits
    cam_infos_unsorted = readToRFCameras(path, tof_extrinsics, tof_intrinsics, color_extrinsics, color_intrinsics, depth_range, znear, zfar, args)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # if not args.dynamic and eval:
    #     if args.train_views != "":
    #         idx_train = [int(i) for i in args.train_views.split(",").strip()]
    #         idx_test = [i for i in np.arange(args.total_num_views) if (i not in idx_train)] 
    #         train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in idx_train]
    #         test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in idx_test]
    #     else:
    #         train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
    #         test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    # else:
    #     train_cam_infos = cam_infos
    #     test_cam_infos = cam_infos
    train_cam_infos = cam_infos
    test_cam_infos = cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if nerf_normalization['radius'] < 1e-6: # camera is fixed, for the synthetic debug cube scene
        # max_depth = np.max([depth_from_tof(train_cam_infos[i].tof_image, depth_range, args.phase_offset) for i in range(args.total_num_views)])
        nerf_normalization['radius'] = 1

    # intrinsics, extrinsics = color_intrinsics + tof_intrinsics, np.concatenate([color_extrinsics, tof_extrinsics], axis=0)
    # poses = [np.linalg.inv(ext) for ext in extrinsics]
    # poses = get_render_poses_spiral(-1.0, np.array([znear, zfar]), intrinsics, poses, nerf_normalization['radius'], N_views=args.total_num_spiral_views)
    # spiral_exts = np.array([np.linalg.inv(pose) for pose in poses])
    # spiral_cam_infos = readToRFSpiralCameras(spiral_exts, color_intrinsics, depth_range, znear, zfar, args)

    # Initialize point cloud
    ply_path = os.path.join(path, "points3d.ply")
    if args.init_method == "random":
        num_pts = args.num_points
        print(f"Generating random point cloud ({num_pts})...")

        # Init xyz
        min_bounds, max_bounds = calculateSceneBounds(train_cam_infos, args)
        xyz = np.random.uniform(min_bounds, max_bounds, (num_pts, 3))
       
        # Init color and phasor
        shs_color = RGB2SH(np.ones((num_pts, 3)) * 0.5)
        colors = SH2RGB(shs_color)
        shs_phase = PA2SH(np.random.random((num_pts, 1)) * 2.0 * np.pi) # The phase will be used if use_view_dependent_phase = True
        shs_amp = PA2SH(np.random.random((num_pts, 1)) * np.sum(np.square(xyz), axis=-1).reshape(num_pts, 1))
        phases = SH2PA(shs_phase) 
        amplitudes = SH2PA(shs_amp)
        motion_mask = np.ones((num_pts, 1)).astype(bool) # All Gaussians are dynamic
    else: # args.init_method == "phase"
        canonical_tof_cam = train_cam_infos[args.canonical_frame_id]
        tof_depth_height = math.ceil(args.tof_image_height / args.phase_resolution_stride * args.tof_scale_factor)
        tof_depth_width = math.ceil(args.tof_image_width / args.phase_resolution_stride * args.tof_scale_factor)

        # Pixel space
        # Static Gaussians
        xy_static = np.indices((tof_depth_height, tof_depth_width)).transpose(1, 2, 0).reshape(-1, 2).astype(np.float32)[:, ::-1] * args.phase_resolution_stride / args.tof_scale_factor
        
        # # Dynamic Gaussians
        # xy_dynamic = np.empty((0, 2))
        # if args.use_motion_mask: #TODO: Improve this part
        #     y_dynamic, x_dynamic = np.where(canonical_tof_cam.motion_mask)
        #     xy_dynamic_all = np.stack([x_dynamic, y_dynamic], axis=-1)
        #     xy_dynamic = xy_dynamic_all[np.random.choice(xy_dynamic_all.shape[0], size=xy_dynamic_all.shape[0]//args.motion_mask_stride, replace=False)]
        #     keep = np.ones_like(xy_static[:, 0]).astype(bool)
        #     for row_idx in range(xy_static.shape[0]):
        #         if np.any(np.all(xy_static[row_idx] == xy_dynamic, axis=1)):
        #             keep[row_idx] = False
        #     xy_static = xy_static[keep]
        # xy_all = np.concatenate([xy_static, xy_dynamic], axis=0).astype(np.int16)
        xy_all = xy_static.astype(np.int16)

        num_pts = xy_all.shape[0]
        print(f"Generating point cloud based on depth from the canonical frame ({num_pts})...")
        
        xyzw = np.empty((num_pts, 4))
        view_mat = getWorld2View2(canonical_tof_cam.R_tof, canonical_tof_cam.T_tof)

        # Normalize to [-WInMeters/2, WInMeters/2] and [-HInMeters/2, HInMeters/2]
        WInMeters = canonical_tof_cam.znear * np.tan(canonical_tof_cam.FovX_tof / 2.0) * 2.0
        HInMeters = canonical_tof_cam.znear * np.tan(canonical_tof_cam.FovY_tof / 2.0) * 2.0

        xyzw[:, 0] = (xy_all[:, 0] * 2.0 / args.tof_image_width - 1.0) * WInMeters / 2.0
        xyzw[:, 1] = (xy_all[:, 1] * 2.0 / args.tof_image_height - 1.0) * HInMeters / 2.0

        # Distances to Light
        z = depth_from_tof(canonical_tof_cam.tof_image[xy_all[:, 1], xy_all[:, 0], :], depth_range, args.phase_offset).reshape(num_pts, 1) 

        # Camera space.
        dists2pixInMeters = np.sqrt(np.square(xyzw[:, 0]) + np.square(xyzw[:, 1]) + np.square(canonical_tof_cam.znear))
        np.true_divide(xyzw[:, 0], dists2pixInMeters, out=xyzw[:, 0]) 
        np.true_divide(xyzw[:, 1], dists2pixInMeters, out=xyzw[:, 1])
        np.multiply(xyzw[:, 0:1], z, out=xyzw[:, 0:1])
        np.multiply(xyzw[:, 1:2], z, out=xyzw[:, 1:2])
        xyzw[:, 2:3] = np.sqrt(np.square(z) - np.square(xyzw[:, 0:1]) - np.square(xyzw[:, 1:2]))
        xyzw[:, 3:4] = np.ones((num_pts, 1))

        # World space.
        xyz = (np.linalg.inv(view_mat) @ xyzw.T).T[:, :3]

        # Motion mask
        motion_mask = np.ones((num_pts, 1)).astype(bool) # All Gaussians are dynamic
        # if args.use_motion_mask: # Separate static (0) and dynamic (1) Gaussians
        #     # motion_mask = np.concatenate([np.zeros_like(xy_static[:, 0]), np.ones_like(movable_xy[:, 0])], axis=0).astype(bool).reshape(num_pts, 1)
        #     motion_mask = (xyz[:, 2] < 2.55).reshape(-1, 1)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=motion_mask, cmap="winter")
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # cbar = plt.colorbar(scatter, ax=ax)
        # cbar.set_label('Mask Value')
        # # plt.show()
        # plt.savefig("w_mm_pcd_viz_init.png" if args.use_motion_mask else "wo_mm_pcd_viz_init.png")
        # plt.close()

        # Init color and phasor
        shs_color = RGB2SH(np.ones((num_pts, 3)) * 0.5)
        colors = SH2RGB(shs_color)
        shs_phase = PA2SH(np.zeros((num_pts, 1)).astype(np.float32))
        shs_amp = PA2SH(canonical_tof_cam.tof_image[xy_all[:, 1], xy_all[:, 0], 2].reshape(-1, 1) * np.square(z))
        phases = SH2PA(shs_phase) 
        amplitudes = SH2PA(shs_amp)

    pcd = BasicPointCloud(points=xyz, colors=colors, normals=np.zeros((num_pts, 3)), phases=phases, amplitudes=amplitudes, motion_mask=motion_mask)

    colors *= 255.0
    storePly(ply_path, xyz, colors)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,  # colmap dataset reader from official 3D Gaussian [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/]
    "Blender": readNerfSyntheticInfo,  # D-NeRF dataset [https://drive.google.com/file/d/1uHVyApwqugXTFuIRRlE4abTW8_rrVeIK/view?usp=sharing]
    "DTU": readNeuSDTUInfo,  # DTU dataset used in Tensor4D [https://github.com/DSaurus/Tensor4D]
    "nerfies": readNerfiesInfo,  # NeRFies & HyperNeRF dataset proposed by [https://github.com/google/hypernerf/releases/tag/v0.1]
    "plenopticVideo": readPlenopticVideoDataset,  # Neural 3D dataset in [https://github.com/facebookresearch/Neural_3D_Video]
    "ToRF": readToRFSceneInfo
}
