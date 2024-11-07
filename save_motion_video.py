from moviepy.editor import ImageSequenceClip
import os

# Directory containing images
image_dir = "/fs/nexus-projects/video-depth-pose/videosfm/test/Deformable-3D-Gaussians/output/tmp"
fps = 30  # Frames per second of the resulting video

# Ensure the images are sorted by frame number
image_files = [os.path.join(image_dir, img) for img in sorted(os.listdir(image_dir)) if img.endswith(".png")]

# Load images into a moviepy clip
clip = ImageSequenceClip(image_files, fps=fps)

# Specify output video file path
output_video_path = os.path.join(image_dir, "output_video.mp4")

# Write the video file to disk
clip.write_videofile(output_video_path, codec='libx264')

print(f"Video compiled and saved to {output_video_path}")
