import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from transformers import CLIPProcessor, CLIPModel

# ====== 1. Load model ======
model_name = "openai/clip-vit-large-patch14"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.eval()

# ====== 2. Đường dẫn root folder ======
ROOT = "./keyframes"   # root chứa nhiều folder con

# ====== 3. Hàm tính embedding cho 1 folder ======
def embed_folder(folder_path, save_path, batch_size=32):
    image_files = sorted(
        [os.path.join(folder_path, f) 
         for f in os.listdir(folder_path) 
         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    )

    embeddings = []
    batch_images = []
    
    for img_path in tqdm(image_files, desc=f"Embedding {os.path.basename(folder_path)}"):
        try:
            image = Image.open(img_path).convert("RGB")
            batch_images.append(image)

            # Nếu đủ batch thì xử lý
            if len(batch_images) == batch_size:
                inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    embs = model.get_image_features(**inputs)
                    embs = embs / embs.norm(p=2, dim=-1, keepdim=True)  # normalize
                    embeddings.append(embs.cpu().numpy())
                batch_images = []

        except Exception as e:
            print(f"⚠️ Lỗi với ảnh {img_path}: {e}")

    # Xử lý batch cuối
    if batch_images:
        inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            embs = model.get_image_features(**inputs)
            embs = embs / embs.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(embs.cpu().numpy())

    if embeddings:
        embeddings = np.concatenate(embeddings, axis=0)
        np.save(save_path, embeddings)
        print(f"✅ Saved {save_path}, shape={embeddings.shape}")
    else:
        print(f"❌ Folder {folder_path} không có ảnh hợp lệ!")

# ====== 4. Chạy cho tất cả folder con ======
output_dir = "./clip-vit-large-patch14"
os.makedirs(output_dir, exist_ok=True)

for folder in os.listdir(ROOT):
    folder_path = os.path.join(ROOT, folder)
    if os.path.isdir(folder_path):
        save_file = os.path.join(output_dir, f"{folder}.npy")
        if not os.path.exists(save_file):
            embed_folder(folder_path, save_file, batch_size=32)  # chỉnh batch_size tuỳ GPU