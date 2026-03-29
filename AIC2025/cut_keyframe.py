import os
import pandas as pd
import cv2
import numpy as np

# Default threshold, can be overridden
DEFAULT_THRESHOLD = 0.95

def get_image_vector(image, size=(64, 64)):
    try:
        image = cv2.resize(image, size)
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()
    except Exception as e:
        print(f'Error computing vector: {e}')
        return np.zeros(512)

def calculate_cosine(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / norm_product

def extract_keyframes(video_path, output_dir, threshold=DEFAULT_THRESHOLD, progress_callback=None):
    if not os.path.exists(video_path):
        print(f'Video path not found: {video_path}')
        return None

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f'Starting processing video: {video_name}')

    keyframes_subdir = os.path.join(output_dir, 'keyframes', video_name)
    map_dir = os.path.join(output_dir, 'map_keyframes')
    
    os.makedirs(keyframes_subdir, exist_ok=True)
    os.makedirs(map_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Cannot open video {video_name}')
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f'FPS of {video_name} is 0. Using default 30.')
        fps = 30
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    map_keyframes_data = []
    
    last_frame_vector = None
    saved_frame_count = 0
    current_frame_idx = 0

    while current_frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            current_frame_idx += int(fps)
            continue

        if last_frame_vector is None:
            is_accepted = True
        else:
            current_frame_vector = get_image_vector(frame)
            sim = calculate_cosine(last_frame_vector, current_frame_vector)
            is_accepted = (sim < threshold)

        if is_accepted:
            pst_time = current_frame_idx / fps
            frame_filename = os.path.join(keyframes_subdir, f'{saved_frame_count:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
            
            map_keyframes_data.append({
                'n': saved_frame_count,
                'pst_time': pst_time,
                'fps': fps,
                'frame_index': current_frame_idx
            })
            
            last_frame_vector = get_image_vector(frame)
            saved_frame_count += 1
        
        # Skip approx 1 second
        frames_to_skip = int(fps)
        current_frame_idx += frames_to_skip if frames_to_skip > 0 else 30
        
        if progress_callback:
            progress_callback(min(current_frame_idx, total_frames), total_frames)

    cap.release()

    if saved_frame_count > 0:
        map_df = pd.DataFrame(map_keyframes_data)
        map_path = os.path.join(map_dir, f'{video_name}.csv')
        map_df.to_csv(map_path, index=False)
        
    print(f'Finished {video_name}: Extracted {saved_frame_count} keyframes.')
    return saved_frame_count
