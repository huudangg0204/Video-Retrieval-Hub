import os
import numpy as np
import faiss
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
import open_clip
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BlipProcessor, BlipForImageTextRetrieval
# 1. Load eva model
model_name = "hf-hub:timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k"
model_eva, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    model_name, pretrained="laion2b_s4b_b131k"
)
tokenizer_eva = open_clip.get_tokenizer(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_eva = model_eva.to(device).eval()

# 2. Load embeddings (npy), build id_to_name mapping as earlier
EMB_PATH = r"./eva02_large_patch14_clip_224.merged2b_s4b_b131k"      # embeddings root
FRAME_PATH = r"./keyframes" # image root directory
EMB_INDEX = r"./eva02_large_patch14_clip_224.merged2b_s4b_b131k.index"
BATCH_IMG = 32
BATCH_TXT = 32
all_embs = []
id_to_name = {}
current_id = 0

print("Loading embeddings from:", EMB_PATH)
print("Image root directory:", FRAME_PATH)
print("Index file path:", EMB_INDEX)

for fname in sorted(os.listdir(EMB_PATH)):
    if fname.endswith(".npy"):
        video_id = os.path.splitext(fname)[0]
        emb = np.load(os.path.join(EMB_PATH, fname))  # shape (N, D)
        all_embs.append(emb)
        for i in range(emb.shape[0]):
            frame_name = f"{i+1:03d}.jpg"
            frame_path = os.path.join(FRAME_PATH, video_id, frame_name)
            id_to_name[current_id] = frame_path
            current_id += 1

all_embs = np.concatenate(all_embs, axis=0).astype("float32")
print("Total vectors:", all_embs.shape)

if not os.path.exists(EMB_INDEX):
    # 3. Build FAISS index (cosine via inner-product after normalization)
    d = all_embs.shape[1]
    index = faiss.IndexFlatIP(d)
    # faiss.normalize_L2(all_embs)
    index.add(all_embs)
    # 7. Save faiss index
    faiss.write_index(index, "./eva02_large_patch14_clip_224.merged2b_s4b_b131k.index")
else:
    index = faiss.read_index(EMB_INDEX)
print("FAISS index size:", index.ntotal)

# 5. (Optional) search_by_text_clip()
def search_by_text_clip(text: str, topk: int = 5):
    text_emb = encode_texts([text])
    q = text_emb.cpu().numpy().astype("float32")
    D, I = index.search(q, topk)
    results = [(id_to_name[idx], float(score)) for idx, score in zip(I[0], D[0])]
    return results



def encode_texts(texts):
    embs = []
    for i in tqdm(range(0, len(texts), BATCH_TXT), desc="Encode texts"):
        tokens = tokenizer_eva(texts[i:i+BATCH_TXT]).to(device)
        with torch.no_grad():
            feats = model_eva.encode_text(tokens)
            feats = F.normalize(feats, dim=-1)
        embs.append(feats.cpu())
    return torch.cat(embs, dim=0)

# 6. Example usage
print("\nText query results:")
text_query = "The athlete"
result = search_by_text_clip(text_query, topk=500)


top500 = sorted(result, key=lambda x: x[1], reverse=True)[:500]


# =====================
# 1. Load model BLIP
# =====================
model_name = "Salesforce/blip-itm-base-coco"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForImageTextRetrieval.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image_paths = []
for path, _ in top500:
    image_paths.append(path)

print(f"Found {len(image_paths)} images")


# =====================
# 4. Tính điểm ITM cho mỗi ảnh
# =====================
scores = []
for idx, path in enumerate(image_paths):
    # try:
    #     image = Image.open(path).convert("RGB")
    # except:
    #     continue
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, text=text_query, return_tensors="pt").to(device)
    with torch.no_grad():
        itm_score = model(**inputs)[0]  # shape [1, 2]
        prob = F.softmax(itm_score, dim=-1)[0, 1]  # xác suất match
        scores.append((path, prob.item()))


# =====================
# 5. Lấy Top-5 ảnh
# =====================
top100 = sorted(scores, key=lambda x: x[1], reverse=True)[:100]

print("\nTop 5 retrieval results:")
for path, score in top100:
    print(f"{path}: {score:.3f}")

# =====================
# 6. Hiển thị Top-5 ảnh
# =====================
plt.figure(figsize=(15, 6))
for i, (path, score) in enumerate(top100[:5]):
    print(path, score)
    img = Image.open(path).convert("RGB")
    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    plt.title(f"{score:.2f}")
    plt.axis("off")

plt.show()
