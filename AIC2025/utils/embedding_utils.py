import os
import torch
import numpy as np
import threading
from PIL import Image
# from tqdm import tqdm # optional
import open_clip
import torch.nn.functional as F

# Adjust import based on your project structure
# If this file is in utils/, then .model_loader is correct
from .model_loader import EVA02Model

def generate_session_embeddings(session_dir, progress_callback=None):
    """
    Generate embeddings for all keyframes in session_dir/keyframes
    Save .npy files to session_dir/embeddings
    """
    keyframes_root = os.path.join(session_dir, 'keyframes')
    embeddings_root = os.path.join(session_dir, 'embeddings')
    
    if not os.path.exists(keyframes_root):
        print(f"Keyframes dir not found: {keyframes_root}")
        return

    os.makedirs(embeddings_root, exist_ok=True)

    # Get singleton model instance
    instance = EVA02Model.get_instance()
    model = instance['model']
    preprocess_val = instance['preprocess_val']
    device = instance['device']
    
    # List video folders in keyframes/
    video_folders = [f for f in os.listdir(keyframes_root) if os.path.isdir(os.path.join(keyframes_root, f))]
    
    # Calculate total images to support accurate progress callbacks
    total_images = 0
    for video_name in video_folders:
        folder = os.path.join(keyframes_root, video_name)
        total_images += len([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    current_processed = [0] # Use list to pass by reference

    for video_name in video_folders:
        video_path = os.path.join(keyframes_root, video_name)
        save_path = os.path.join(embeddings_root, f"{video_name}.npy")
        
        # logical check: if exists, maybe skip
        if os.path.exists(save_path):
            current_processed[0] += len([f for f in os.listdir(video_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if progress_callback:
                progress_callback(current_processed[0], total_images)
            continue
            
        process_single_folder(video_path, save_path, model, preprocess_val, device, 
                            current_processed=current_processed, 
                            total_images=total_images, 
                            progress_callback=progress_callback)

def process_single_folder(folder_path, save_path, model, preprocess_val, device, batch_size=32, current_processed=None, total_images=0, progress_callback=None):
    image_files = sorted(
        [os.path.join(folder_path, f) 
         for f in os.listdir(folder_path) 
         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    )

    if not image_files:
        return

    embeddings = []
    batch_images = []
    
    print(f"Embedding {os.path.basename(folder_path)} with {len(image_files)} images")

    for img_path in image_files:
        try:
            image = Image.open(img_path).convert("RGB")
            # Preprocess
            batch_images.append(preprocess_val(image))

            if len(batch_images) == batch_size:
                batch_tensor = torch.stack(batch_images).to(device)
                with torch.no_grad():
                    feats = model.encode_image(batch_tensor)
                    feats = F.normalize(feats, dim=-1)
                embeddings.append(feats.cpu().numpy())
                batch_images = []
                
                if current_processed is not None and progress_callback:
                    current_processed[0] += batch_size
                    progress_callback(current_processed[0], total_images)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Process remaining
    if batch_images:
        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
            feats = F.normalize(feats, dim=-1)
        embeddings.append(feats.cpu().numpy())
        
        if current_processed is not None and progress_callback:
            current_processed[0] += len(batch_images)
            progress_callback(current_processed[0], total_images)

    if embeddings:
        final_embeddings = np.concatenate(embeddings, axis=0)
        np.save(save_path, final_embeddings)
        print(f"Saved embeddings to {save_path}, shape={final_embeddings.shape}")
