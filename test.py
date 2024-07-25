import numpy as np
import os
import matplotlib.pyplot as plt
import imageio

data_root = "/fs/nexus-projects/video-depth-pose/videosfm/datasets/torf_data/deskbox/deskbox/depth"

render_root = "/fs/nexus-projects/video-depth-pose/videosfm/test/Deformable-3D-Gaussians/output/deskbox/train/ours_20000/depth"

gt_depth = np.load(os.path.join(data_root, "0001.npy"))
rendered_depth = imageio.imread(os.path.join(render_root, "00000.png"))

def normalize_im(im):
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    im[np.isnan(im)] = 0.
    return np.clip(im, 0, 1)

def normalize_im_gt(im, im_gt):
    im = (im - np.min(im_gt)) / (np.max(im_gt) - np.min(im_gt))
    im[np.isnan(im)] = 0.
    return np.clip(im, 0, 1)

plt.imsave("gt_depth_0.png", normalize_im(gt_depth), cmap="gray")
print(gt_depth.shape)

plt.imsave("rendered_depth_0.png", rendered_depth)
print(rendered_depth.shape)