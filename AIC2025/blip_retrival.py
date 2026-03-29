import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForImageTextRetrieval
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
# =====================
# 1. Load model
# =====================
model_name = "Salesforce/blip-itm-base-coco"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForImageTextRetrieval.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# =====================
# 2. Thư viện ảnh (có nhiều folder con)
# =====================
image_root = r"./keyframes"
max_image = 500
valid_exts = {".jpg", ".jpeg", ".png"}
all_image_paths = []
for root, _, files in os.walk(image_root):
    for f in files:
        if os.path.splitext(f)[1].lower() in valid_exts:
            all_image_paths.append(os.path.join(root, f))

# Limit to max_image
image_paths = all_image_paths[:min(max_image, len(all_image_paths))]

print(f"Found {len(all_image_paths)} images, using {len(image_paths)} for retrieval")

# =====================
# 3. Query text
# =====================

text_query = "a man wearing a wide-brimmed hat and a blue shirt"

# =====================
# 4. Tính điểm ITM cho mỗi ảnh
# =====================
scores = []
for path in tqdm(image_paths, desc="Processing images", unit="img"):
    try:
        image = Image.open(path).convert("RGB")
    except:
        continue

    inputs = processor(images=image, text=text_query, return_tensors="pt").to(device)
    with torch.no_grad():
        itm_score = model(**inputs)[0]  # shape [1, 2]
        prob = F.softmax(itm_score, dim=-1)[0, 1]  # xác suất match
        scores.append((path, prob.item()))

# =====================
# 5. Lấy Top-5 ảnh
# =====================
top5 = sorted(scores, key=lambda x: x[1], reverse=True)[:5]

print("\nTop 5 retrieval results:")
for path, score in top5:
    print(f"{path}: {score:.4f}")

# =====================
# 6. Hiển thị Top-5 ảnh
# =====================
plt.figure(figsize=(15, 6))
for i, (path, score) in enumerate(top5):
    img = Image.open(path).convert("RGB")
    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    plt.title(f"{score:.2f}")
    plt.axis("off")
plt.show()


