import cv2
import os
import sys
from pathlib import Path

# Define paths - accept command line arguments
max_frames = None
if len(sys.argv) > 1:
    video_arg = sys.argv[1]  # e.g., "fall5", "fall5.avi", or "lab_cap1.mp4"
    video_stem = Path(video_arg).stem
    video_suffix = Path(video_arg).suffix.lower()
    if video_suffix:
        video_path = f"surface_contact_exp/videos/{video_arg}"
    else:
        avi_path = Path(f"surface_contact_exp/videos/{video_arg}.avi")
        mp4_path = Path(f"surface_contact_exp/videos/{video_arg}.mp4")
        video_path = str(avi_path if avi_path.exists() else mp4_path)
    output_dir = f"surface_contact_exp/images/{video_stem}_images"
    if len(sys.argv) > 2:
        max_frames = int(sys.argv[2])
else:
    video_path = "surface_contact_exp/videos/fall4.avi"
    output_dir = "surface_contact_exp/images/fall4_images"

# Check if video exists
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    exit(1)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit(1)

# Get video properties
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video properties:")
print(f"  Total frames: {frame_count}")
print(f"  FPS: {fps}")
print(f"  Resolution: {width}x{height}")
print(f"\nExtracting frames to {output_dir}...")

# Extract frames
frame_num = 0
while True:
    if max_frames is not None and frame_num >= max_frames:
        break

    ret, frame = cap.read()
    if not ret:
        break
    
    # Save frame as image
    output_path = os.path.join(output_dir, f"frame_{frame_num:06d}.jpg")
    cv2.imwrite(output_path, frame)
    
    if (frame_num + 1) % 50 == 0:
        print(f"  Extracted {frame_num + 1}/{frame_count} frames")
    
    frame_num += 1

cap.release()
print(f"\nCompleted! Extracted {frame_num} frames to {output_dir}")
