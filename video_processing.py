# import os
# import random
# import cv2
# import numpy as np
# from concurrent.futures import ThreadPoolExecutor, as_completed

# def extract_frames(video_path, output_dir, interval=30):
#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if frame_count % interval == 0:
#             frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
#             cv2.imwrite(frame_path, frame)
#         frame_count += 1
#     cap.release()

# def process_video(video_path, output_video_dir, frame_interval=30):
#     video_name = os.path.splitext(os.path.basename(video_path))[0]
#     relative_path = os.path.relpath(os.path.dirname(video_path), input_directory)

#     # Create output directory for frames
#     video_output_dir = os.path.join(output_video_dir, relative_path, video_name)
#     if os.path.exists(video_output_dir) and len(os.listdir(video_output_dir)) > 0:
#         print(f"Skipping {video_name}, already processed.")
#         return

#     os.makedirs(video_output_dir, exist_ok=True)

#     # Extract frames only
#     extract_frames(video_path, video_output_dir, frame_interval)

# def preprocess_videos(input_dir, output_video_dir, frame_interval=30, sample_ratio=1.0):
#     video_files = []
#     for root, dirs, files in os.walk(input_dir):
#         for file in files:
#             if file.endswith('.mp4'):
#                 video_files.append(os.path.join(root, file))

#     sample_size = int(len(video_files) * sample_ratio)
#     sampled_videos = random.sample(video_files, sample_size)

#     with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
#         futures = [
#             executor.submit(process_video, video, output_video_dir, frame_interval)
#             for video in sampled_videos
#         ]
#         for future in as_completed(futures):
#             try:
#                 future.result()
#             except Exception as e:
#                 print(f"Error processing video: {e}")

# # Define directories
# input_directory = "videos"
# output_video_dir = "preprocessed_videos"

# # Create output directory if it doesn't exist
# os.makedirs(output_video_dir, exist_ok=True)

# # Preprocess all videos
# preprocess_videos(input_directory, output_video_dir, sample_ratio=1.0)


import os
import random
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_frames(video_path, output_dir, interval=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        frame_count += 1
    cap.release()

def process_video(video_path, output_video_dir, frame_interval=30):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Get class (real/fake) from parent directory
    class_folder = os.path.basename(os.path.dirname(video_path))
    video_output_dir = os.path.join(output_video_dir, class_folder, video_name)
    if os.path.exists(video_output_dir) and len(os.listdir(video_output_dir)) > 0:
        print(f"Skipping {video_name}, already processed.")
        return
    os.makedirs(video_output_dir, exist_ok=True)
    extract_frames(video_path, video_output_dir, frame_interval)

def preprocess_videos(input_dir, output_video_dir, frame_interval=30, sample_ratio=1.0):
    video_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_files.append(os.path.join(root, file))
    sample_size = int(len(video_files) * sample_ratio)
    sampled_videos = random.sample(video_files, sample_size) if sample_ratio < 1.0 else video_files
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(process_video, video, output_video_dir, frame_interval)
            for video in sampled_videos
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing video: {e}")

if __name__ == "__main__":
    input_directory = "videos"
    output_video_dir = "preprocessed_videos"
    os.makedirs(output_video_dir, exist_ok=True)
    preprocess_videos(input_directory, output_video_dir, sample_ratio=1.0)
