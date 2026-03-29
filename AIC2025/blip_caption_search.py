"""
Search Engine from BLIP Image Captioning (Salesforce/blip-image-captioning-large)

Features
- Index a folder of images by first captioning each image with BLIP, then embedding captions.
- Vector backends: FAISS (if installed), scikit-learn NearestNeighbors, or a pure NumPy fallback.
- Query by TEXT or by IMAGE (the image is first captioned, then searched).
- Saves/loads the index so you don't need to re-caption every time.

Quickstart
----------
# 1) Install deps (choose one vector backend; FAISS is optional)
#   pip install "transformers>=4.41" "torch>=2.1" pillow sentence-transformers tqdm
#   pip install faiss-cpu           # optional, recommended for large indexes
#   pip install scikit-learn        # optional, for NearestNeighbors backend

# 2) Build an index from an image folder
#   python blip_caption_search.py build \
#       --images_dir /path/to/images \
#       --workdir ./blip_index \
#       --backend faiss \
#       --caption_batch 4

python .\blip_caption_search.py build --images_dir .\keyframes\ --workdir .\blip_index\ --caption_batch 8 

# 3) Search by text
#   python blip_caption_search.py search-text \
#       --workdir ./blip_index \
#       --query "cat sitting on a wooden chair" \
#       --topk 5

# 4) Or search by image (the query image is captioned first)
#   python blip_caption_search.py search-image \
#       --workdir ./blip_index \
#       --image /path/to/query.jpg \
#       --topk 5

Notes
- The BLIP model is large; ensure you have enough VRAM/RAM. It will run on CPU if CUDA isn't available (slower).
- The caption quality drives retrieval quality; you can adjust generation settings below.
- For multilingual text search, switch the sentence-transformers model to a multilingual one (e.g., paraphrase-multilingual-MiniLM-L12-v2).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from transformers import AutoProcessor
from transformers import BlipForConditionalGeneration

# Text embedding (you can swap to another sentence-transformers model if desired)
from sentence_transformers import SentenceTransformer

from tqdm import tqdm

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False

try:
    from sklearn.neighbors import NearestNeighbors  # type: ignore
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False


# -----------------------------
# Utilities
# -----------------------------

def find_images(root: Path, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")) -> List[Path]:
    files = []
    for p in root.rglob('*'):
        if p.suffix.lower() in exts and p.is_file():
            files.append(p)
    return files


def normalize(vecs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return vecs / norms


# -----------------------------
# Captioner
# -----------------------------
@dataclass
class BLIPCaptioner:
    model_name: str = "Salesforce/blip-image-captioning-large"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 30
    num_beams: int = 3  # use >1 for better captions, at cost of speed

    def __post_init__(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def caption_image(self, image: Image.Image) -> str:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
        )
        caption = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        return caption.strip()

    def caption_paths(self, paths: List[Path], batch: int = 4) -> List[str]:
        captions: List[str] = []
        for i in tqdm(range(0, len(paths), batch), desc="Captioning"):
            batch_paths = paths[i:i+batch]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            with torch.inference_mode():
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                )
                batch_caps = self.processor.batch_decode(out, skip_special_tokens=True)
                captions.extend([c.strip() for c in batch_caps])
            # close PIL images
            for im in images:
                im.close()
        return captions


# -----------------------------
# Text Embedder
# -----------------------------
@dataclass
class TextEmbedder:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        embs = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
        return embs.astype(np.float32)


# -----------------------------
# Vector Index Backends
# -----------------------------
class VectorBackend:
    def build(self, vectors: np.ndarray):
        raise NotImplementedError

    def search(self, query: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def save(self, path: Path):
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path):
        raise NotImplementedError


class FaissBackend(VectorBackend):
    def __init__(self):
        if not _FAISS_AVAILABLE:
            raise RuntimeError("FAISS is not installed. Try 'pip install faiss-cpu' or use backend 'nn' or 'numpy'.")
        self.index = None

    def build(self, vectors: np.ndarray):
        d = vectors.shape[1]
        self.index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)

    def search(self, query: np.ndarray, topk: int):
        faiss.normalize_L2(query)
        D, I = self.index.search(query, topk)
        return D, I

    def save(self, path: Path):
        assert self.index is not None
        faiss.write_index(self.index, str(path / "index.faiss"))

    @classmethod
    def load(cls, path: Path):
        obj = cls()
        obj.index = faiss.read_index(str(path / "index.faiss"))
        return obj


class NNBackend(VectorBackend):
    def __init__(self, n_neighbors: int = 10):
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not installed. Try 'pip install scikit-learn' or use another backend.")
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        self.vectors = None

    def build(self, vectors: np.ndarray):
        self.vectors = normalize(vectors)
        self.nn.set_params(n_neighbors=min(self.nn.n_neighbors, len(vectors)))
        self.nn.fit(self.vectors)

    def search(self, query: np.ndarray, topk: int):
        q = normalize(query)
        distances, idx = self.nn.kneighbors(q, n_neighbors=min(topk, self.nn.n_neighbors), return_distance=True)
        # convert cosine distance -> cosine similarity
        sims = 1.0 - distances
        return sims.astype(np.float32), idx.astype(np.int64)

    def save(self, path: Path):
        # store vectors for exact search at load-time
        if self.vectors is None:
            raise RuntimeError("No vectors to save.")
        np.save(path / "vectors.npy", self.vectors)

    @classmethod
    def load(cls, path: Path):
        obj = cls()
        obj.vectors = np.load(path / "vectors.npy")
        obj.nn.fit(obj.vectors)
        return obj


class NumpyBackend(VectorBackend):
    def __init__(self):
        self.vectors = None

    def build(self, vectors: np.ndarray):
        self.vectors = normalize(vectors)

    def search(self, query: np.ndarray, topk: int):
        q = normalize(query)
        sims = q @ self.vectors.T  # cosine similarity after normalization
        idx = np.argsort(-sims, axis=1)[:, :topk]
        top_sims = np.take_along_axis(sims, idx, axis=1)
        return top_sims.astype(np.float32), idx.astype(np.int64)

    def save(self, path: Path):
        if self.vectors is None:
            raise RuntimeError("No vectors to save.")
        np.save(path / "vectors.npy", self.vectors)

    @classmethod
    def load(cls, path: Path):
        obj = cls()
        obj.vectors = np.load(path / "vectors.npy")
        return obj


BACKENDS = {
    "faiss": FaissBackend,
    "nn": NNBackend,
    "numpy": NumpyBackend,
}


# -----------------------------
# Search Engine
# -----------------------------
@dataclass
class ImageSearchEngine:
    workdir: Path
    backend_name: str = "faiss"
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.workdir / "meta.json"
        self.backend_path = self.workdir / "backend"
        self.backend_path.mkdir(exist_ok=True)
        self.checkpoint_dir = self.workdir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.embedder = TextEmbedder(self.text_model_name, self.device)
        self.backend: Optional[VectorBackend] = None
    def _init_backend(self) -> VectorBackend:
        if self.backend_name not in BACKENDS:
            raise ValueError(f"Unknown backend '{self.backend_name}'. Available: {list(BACKENDS.keys())}")
        return BACKENDS[self.backend_name]()
    # ---------- Build ----------
    def build_from_folder(self, images_dir: Path, caption_batch: int = 4,
                          blip_model: str = "Salesforce/blip-image-captioning-large",
                          max_new_tokens: int = 30, num_beams: int = 3):
        cap = BLIPCaptioner(model_name=blip_model, device=self.device,
                            max_new_tokens=max_new_tokens, num_beams=num_beams)

        # group by subdir
        all_paths = find_images(images_dir)
        if not all_paths:
            raise SystemExit(f"No images found under {images_dir}")

        grouped = {}
        for p in all_paths:
            rel = p.relative_to(images_dir)
            subdir = rel.parent if rel.parent != Path('.') else Path('root')
            grouped.setdefault(str(subdir), []).append(p)

        all_records = []
        all_vecs = []

        for subdir, paths in grouped.items():
            cp_dir = self.checkpoint_dir / subdir.replace(os.sep, "_")
            cp_meta = cp_dir / "meta.json"
            cp_vecs = cp_dir / "vectors.npy"

            if cp_meta.exists() and cp_vecs.exists():
                print(f"[Checkpoint] Skipping {subdir}, already processed.")
                recs = [json.loads(line) for line in cp_meta.read_text(encoding='utf-8').splitlines()]
                vecs = np.load(cp_vecs)
                all_records.extend(recs)
                all_vecs.append(vecs)
                continue

            print(f"[Processing] Subdir: {subdir} with {len(paths)} images")
            start = time.time()
            captions = cap.caption_paths(paths, batch=caption_batch)
            print(f"  -> Captioned {len(captions)} images in {time.time()-start:.1f}s")

            # metadata for this subdir
            recs = []
            for i, (p, c) in enumerate(zip(paths, captions)):
                rec = {"id": len(all_records)+i, "image_path": str(p.resolve()), "caption": c}
                recs.append(rec)

            # embed
            print("  -> Embedding captions...")
            vecs = self.embedder.encode(captions)

            # save checkpoint
            cp_dir.mkdir(parents=True, exist_ok=True)
            with cp_meta.open('w', encoding='utf-8') as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            np.save(cp_vecs, vecs)
            print(f"  -> Saved checkpoint to {cp_dir}")

            all_records.extend(recs)
            all_vecs.append(vecs)

        # merge all checkpoints
        with self.meta_path.open('w', encoding='utf-8') as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        X = np.concatenate(all_vecs, axis=0)

        print(f"[Build] Total {len(all_records)} images indexed.")
        self.backend = self._init_backend()
        self.backend.build(X)
        self.backend.save(self.backend_path)
        print(f"[Build] Index built with backend='{self.backend_name}' and saved to {self.backend_path}")

    # ---------- Load ----------
    def load(self):
        if not self.meta_path.exists():
            raise SystemExit(f"Missing metadata {self.meta_path}. Build the index first.")
        self.records = [json.loads(line) for line in self.meta_path.read_text(encoding='utf-8').splitlines()]
        self.backend = BACKENDS[self.backend_name].load(self.backend_path)
        print(f"Loaded {len(self.records)} records and backend '{self.backend_name}'.")

    # ---------- Query ----------
    def _postprocess(self, idx: np.ndarray, sims: np.ndarray, topk: int) -> List[dict]:
        out = []
        for i in range(min(topk, idx.shape[1])):
            rid = int(idx[0, i])
            rec = self.records[rid]
            out.append({
                "rank": i + 1,
                "score": float(sims[0, i]),
                "image_path": rec["image_path"],
                "caption": rec["caption"],
            })
        return out

    def search_by_text(self, query: str, topk: int = 5) -> List[dict]:
        if self.backend is None:
            self.load()
        q = self.embedder.encode([query])  # (1, d)
        sims, idx = self.backend.search(q, topk)
        return self._postprocess(idx, sims, topk)

    def search_by_image(self, image_path: Path, topk: int = 5,
                        blip_model: str = "Salesforce/blip-image-captioning-large",
                        max_new_tokens: int = 30, num_beams: int = 3) -> List[dict]:
        if self.backend is None:
            self.load()
        cap = BLIPCaptioner(model_name=blip_model, device=self.device,
                            max_new_tokens=max_new_tokens, num_beams=num_beams)
        with Image.open(image_path).convert("RGB") as im:
            query_caption = cap.caption_image(im)
        print(f"Query image caption: {query_caption}")
        return self.search_by_text(query_caption, topk=topk)

# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Search engine built on BLIP captions + text embeddings")
    sub = p.add_subparsers(dest="cmd", required=True)

    # build
    b = sub.add_parser("build", help="Caption a folder and build an index")
    b.add_argument("--images_dir", type=str, required=True)
    b.add_argument("--workdir", type=str, required=True)
    b.add_argument("--backend", type=str, default="faiss", choices=list(BACKENDS.keys()))
    b.add_argument("--caption_batch", type=int, default=4)
    b.add_argument("--blip_model", type=str, default="Salesforce/blip-image-captioning-large")
    b.add_argument("--blip_max_new_tokens", type=int, default=30)
    b.add_argument("--blip_num_beams", type=int, default=3)
    b.add_argument("--text_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")

    # search-text
    st = sub.add_parser("search-text", help="Search by a text query")
    st.add_argument("--workdir", type=str, required=True)
    st.add_argument("--backend", type=str, default="faiss", choices=list(BACKENDS.keys()))
    st.add_argument("--text_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    st.add_argument("--query", type=str, required=True)
    st.add_argument("--topk", type=int, default=5)

    # search-image
    si = sub.add_parser("search-image", help="Search by an image (caption the query image first)")
    si.add_argument("--workdir", type=str, required=True)
    si.add_argument("--backend", type=str, default="faiss", choices=list(BACKENDS.keys()))
    si.add_argument("--text_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    si.add_argument("--image", type=str, required=True)
    si.add_argument("--topk", type=int, default=5)
    si.add_argument("--blip_model", type=str, default="Salesforce/blip-image-captioning-large")
    si.add_argument("--blip_max_new_tokens", type=int, default=30)
    si.add_argument("--blip_num_beams", type=int, default=3)

    return p.parse_args()


def main():
    args = parse_args()
    workdir = Path(args.workdir)

    if args.cmd == "build":
        eng = ImageSearchEngine(workdir=workdir, backend_name=args.backend, text_model_name=args.text_model)
        eng.build_from_folder(
            images_dir=Path(args.images_dir),
            caption_batch=args.caption_batch,
            blip_model=args.blip_model,
            max_new_tokens=args.blip_max_new_tokens,
            num_beams=args.blip_num_beams,
        )

    elif args.cmd == "search-text":
        eng = ImageSearchEngine(workdir=workdir, backend_name=args.backend, text_model_name=args.text_model)
        eng.load()
        results = eng.search_by_text(args.query, topk=args.topk)
        print(json.dumps(results, ensure_ascii=False, indent=2))

    elif args.cmd == "search-image":
        eng = ImageSearchEngine(workdir=workdir, backend_name=args.backend, text_model_name=args.text_model)
        eng.load()
        results = eng.search_by_image(Path(args.image), topk=args.topk,
                                      blip_model=args.blip_model,
                                      max_new_tokens=args.blip_max_new_tokens,
                                      num_beams=args.blip_num_beams)
        print(json.dumps(results, ensure_ascii=False, indent=2))

    else:
        raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()
