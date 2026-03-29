import os
import pandas as pd
import cv2
import random   
VIDEO_FOLDER_PATH = "./videos/L26"
KEY_FRAMES_PATH = "keyframes"
MAP_KEYFRAMES_PATH = "map_keyframes"
OUTPUT_FOLDER = "./test_data"

def check_fps(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def extract_key_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # get video name from path
    video_name = os.path.basename(video_path).split('.')[0]
    
    # Create video-specific folder
    video_output_folder = os.path.join(output_folder, KEY_FRAMES_PATH, video_name)
    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)
    
    map_keyframes = pd.DataFrame(columns=["n", "pst_time", "fps", "frame_index"])
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fps = check_fps(video_path)
    map_keyframes["fps"] = fps
    # Get total frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize current frame position
    current_frame = 0
    
    while current_frame < total_frames:
        # Set the video to the current frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate position time in seconds
        pst_time = current_frame / fps
        
        # Save the frame
        frame_filename = os.path.join(video_output_folder, f"{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        map_keyframes.loc[frame_count] = [frame_count, pst_time, fps, current_frame]
        
        frame_count += 1
        
        # Randomly select a delay between 3-5 seconds
        delay_seconds = random.uniform(3, 5)
        frames_to_skip = int(delay_seconds * fps)
        
        # Update current frame position
        current_frame += frames_to_skip
    
    cap.release()
    # save the mapping DataFrame to a xlsx file
    if not os.path.exists(os.path.join(output_folder, MAP_KEYFRAMES_PATH)):
        os.makedirs(os.path.join(output_folder, MAP_KEYFRAMES_PATH))
    map_keyframes.to_csv(os.path.join(output_folder, MAP_KEYFRAMES_PATH, "{}.csv").format(video_name), index=False)
    print(f"Extracted {frame_count} frames from {video_path} to {output_folder}")
import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


def extract_from_videos(video_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4.webm')]
    
    # Function to process a single video
    def process_video(video_file):
        video_path = os.path.join(video_folder, video_file)
        extract_key_frames(video_path, output_folder)

    # Use ThreadPoolExecutor to process videos in parallel

    max_workers = 2
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm.tqdm(executor.map(process_video, video_files), 
                        total=len(video_files), 
                        desc="Processing videos"))

extract_from_videos(VIDEO_FOLDER_PATH, OUTPUT_FOLDER)