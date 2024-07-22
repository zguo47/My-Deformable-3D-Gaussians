import numpy as np
import torch
import torch.nn as nn
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.time_utils import DeformNetwork
from plyfile import PlyData, PlyElement
import os
import time
import json
from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_model(model_path):
    plydata = PlyData.read(model_path)
    print("plydata", plydata)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    # Box center and size
    center_x, center_y, center_z = -1.5, -1, 0
    width, height, depth = 4, 2, 4

    # Calculate boundaries
    x_min, x_max = center_x - width/2, center_x + width/2
    y_min, y_max = center_y - height/2, center_y + height/2
    z_min, z_max = center_z - depth/2, center_z + depth/2

    # Create motion mask for the box
    # motion_mask = (xyz[:, 0] >= x_min) & (xyz[:, 0] <= x_max) & \
    #               (xyz[:, 1] >= y_min) & (xyz[:, 1] <= y_max) & \
    #               (xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max)
    motion_mask = np.ones(len(xyz), dtype=bool)
    np.random.seed(42)

    filtered_indices = np.where(motion_mask)[0]
    selected_indices = np.random.choice(filtered_indices, size=int(len(filtered_indices) * 0.01), replace=False)
    
    refined_motion_mask = np.zeros_like(motion_mask, dtype=bool)
    refined_motion_mask[selected_indices] = True
    
    xyz = xyz[::, :]
    _xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda"))

    return xyz, _xyz, refined_motion_mask

iteration = 10000
model_path = "/fs/nexus-projects/video-depth-pose/videosfm/test/Deformable-3D-Gaussians/output/chicken"
pcd_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
deform_model_path = os.path.join(model_path, "deform", f"iteration_{iteration}", "deform.pth")

xyz_np, xyz_nn, motion_mask = load_model(pcd_path)
deform_model = DeformNetwork(D=8, W=256, multires=10).cuda()
deform_model.load_state_dict(torch.load(deform_model_path), strict=False)

num_frames_to_viz = 81
total_num_frames = 81
canonical_frame_id = 39

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Gaussian Movement Visualization')

# Directory for saving the images
image_dir = "/fs/nexus-projects/video-depth-pose/videosfm/test/Deformable-3D-Gaussians/output/chicken_vis"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Initialize a list to store the trajectory (path) of each point
trajectories = []

for vid in range(num_frames_to_viz):
    t = torch.tensor(np.array([vid / (total_num_frames - 1)])).float().cuda().unsqueeze(0).expand(xyz_np[motion_mask].shape[0], -1)
    delta_xyz, _, _ = deform_model(xyz_nn[motion_mask], t)
    delta_xyz = delta_xyz.detach().cpu().numpy()
    current_frame = delta_xyz + xyz_np[motion_mask]

    # Determine colors for current points based on their 'y' values
    colors = ['red' if y < 0.5 else 'green' for y, z in zip(current_frame[:, 1], current_frame[:, 2])]

    rotation_matrix = np.array([[1, 0, 0],
                            [0, 0, -1],
                            [0, 1, 0]])
    rotated_frame = np.dot(current_frame, rotation_matrix)

    if vid == 0:
        # Initialize trajectories with the first frame positions
        trajectories = [[pos] for pos in rotated_frame]
    else:
        # Append new positions to each trajectory
        for traj, new_pos in zip(trajectories, rotated_frame):
            traj.append(new_pos)

    ax.clear()
    # Plot trajectories
    for traj, color in zip(trajectories, colors):
        x_vals, y_vals, z_vals = zip(*traj)
        # Use a lighter shade of the current point color for the trajectory
        # line_color = 'orange' if color == 'green' else 'blue'
        line_color = 'blue' 
        ax.plot(x_vals, y_vals, z_vals, color=line_color, linewidth=1.0)

    # Plot current points with their respective colors

    ax.scatter(rotated_frame[:, 0], rotated_frame[:, 1], rotated_frame[:, 2], c='r', marker='o', s=0.1)

    # ax.invert_zaxis()
    # ax.invert_xaxis()

    # ax.set_xlim([-0.35, 0.35])
    # ax.set_ylim([-0.3, 0.05])
    # ax.set_zlim([-0.35, 0.35])

    # Save the frame
    plt.savefig(f"{image_dir}/frame_{vid:04d}.png")
    plt.draw()
    plt.pause(0.01)  # pause to allow update of the plot

plt.show()