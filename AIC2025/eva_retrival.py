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

# 1. Load CLIP model & processor
model_name = "hf-hub:timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k"
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    model_name, pretrained="laion2b_s4b_b131k"
)
tokenizer = open_clip.get_tokenizer(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()


# 2. Load embeddings (npy), build id_to_name mapping as earlier
EMB_PATH = f"./eva02_large_patch14_clip_224.merged2b_s4b_b131k-ae"      # embeddings root
FRAME_PATH = "./keyframes-ae" # image root directory
EMB_INDEX = f"./{model_name.split('/')[-1]}-ae.index"
BATCH_IMG = 32
BATCH_TXT = 32
all_embs = []
id_to_name = {}
current_id = 0

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
    faiss.write_index(index, EMB_INDEX)
else:
    index = faiss.read_index(EMB_INDEX)
print("FAISS index size:", index.ntotal)

# 4. Define search_by_image_clip()
def search_by_image_clip(image_path: str, topk: int = 5):
    img = Image.open(image_path).convert("RGB")
    img = preprocess_val(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = model.encode_image(img)
        feats = F.normalize(feats, dim=-1)

    q = feats.cpu().numpy().astype("float32")
    D, I = index.search(q, topk)
    results = [(id_to_name[idx], float(score)) for idx, score in zip(I[0], D[0])]
    return results

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
        tokens = tokenizer(texts[i:i+BATCH_TXT]).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = F.normalize(feats, dim=-1)
        embs.append(feats.cpu())
    return torch.cat(embs, dim=0)

# 6. Example usage
if __name__ == "__main__":
    print("Image query results:")

    print(search_by_image_clip("./000001.jpg", topk=10))

    print("\nText query results:")
    result = search_by_text_clip("the women is wearing red t-shirt in the central of the image.A group of university students sitting in an auditorium, attentively watching a musical performance on stage. The musical play is organized to spread a message about environmental protection. The students appear focused and engaged, with serious expressions, while the atmosphere highlights the importance of raising awareness about protecting nature. ", topk=100)
    # ===================== 
    # 5. Lấy Top-100 ảnh với score > 0.2
    # =====================
    top100 = [item for item in result if item[1] > 0.2]
    top100 = sorted(top100, key=lambda x: x[1], reverse=True)[:100]

    print("\nTop 100 retrieval results:")
    for path, score in top100:
        print(f"{path}: {score:.4f}")



