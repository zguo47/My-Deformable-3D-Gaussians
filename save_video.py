import cv2
import os
from natsort import natsorted

def make_video(image_folder, video_path, fps=15):
    """
    Create a video from a folder of images.

    Args:
    image_folder (str): Path to the folder containing images.
    video_path (str): Path where the video will be saved.
    fps (int, optional): Frames per second for the video. Default is 30.
    """
    # Get all files from the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    # Sort files by name
    images = natsorted(images)

    # Read the first image to obtain width and height
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

# Example usage:
image_folder = '/fs/nexus-projects/video-depth-pose/videosfm/test/Deformable-3D-Gaussians/output/cross_hands/test/ours_20000/renders'  # Path to the folder containing images
video_path = '/fs/nexus-projects/video-depth-pose/videosfm/test/Deformable-3D-Gaussians/output/cross_hands_video_test.mp4'  # Path where the video should be saved
make_video(image_folder, video_path)

