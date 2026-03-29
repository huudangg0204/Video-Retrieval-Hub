# Task 1: Refactor AI Core Components & User Session Storage

## Context
Currently, our retrieval engine (`eva02_retrieval_trake.py`) is designed as a static monolithic system that loads all embeddings from a fixed global directory into memory upon server startup. It is hardcoded to use global variables (like `EMBEDDING_DIR` and `KEYFRAMES_DIR`). 
Our new product objective is a dynamic SaaS-like system: users upload their own videos, and we process and search against those specific videos. We need to modularize the AI scripts (`cut_keyframe.py`, embedding extraction, and FAISS indexing) so they can process independent data folders per user session.

## Requirements & Goals
1. **Dynamic Workspace Generation:** System needs a utility to generate unique workspace folders for every incoming video request (e.g., `/userdata/<session_id>/`).
2. **Modular Keyframe Extractor:** Refactor `cut_keyframe.py` into a reusable module/function that accepts a source video path and a dynamic destination path, rather than using hardcoded paths.
3. **Modular Embedding Generator:** Decouple the embedding logic from `EVA02ImageRetrieval` or `get_embedding.py`. Create an `Embedder` class that evaluates a specific user's `keyframes` folder and outputs `.npy` features.
4. **Dynamic FAISS Manager:** Modify `VectorDatabase` / FAISS logic to create, save, and load a specific `.index` file from the user's specific directory, rather than a global `faiss_db.pkl`.

## Action Plan (Step-by-Step)

### Step 1.1: Define User Directory Structure
Create a standard utility function (e.g., inside a new `utils/storage.py`) to initialize user workspace.
```text
/userdata
    /{session_id}
        /videos          <-- User uploaded raw .mp4
        /keyframes       <-- Output of keyframe extraction
        /map_keyframes   <-- CSV mapping of frames
        /embeddings      <-- Output .npy files
        /db              <-- FAISS index for this session
```

### Step 1.2: Refactor `cut_keyframe.py`
- Wrap the extraction logic in a class/function `extract_keyframes(video_path, output_session_dir)`.
- Ensure it successfully populates the `keyframes` and `map_keyframes` folders within the provided `output_session_dir`.
- Return metadata (e.g., number of frames, video length) to the caller.

### Step 1.3: Refactor Embedding Logic
- Extract the EVA02 model loading logic (currently in `__init__` of `eva02_retrieval_trake.py`) into a singleton or shared instance (so we don't load the large model into memory more than once across concurrent user sessions).
- Create a function `generate_embeddings(session_id, keyframes_dir, embeddings_dir)` that runs the EVA02 inference over the user's keyframes.

### Step 1.4: Refactor FAISS Database Configuration
- Refactor the `VectorDatabase` class. It must take `db_path` as a parameter dynamically based on `session_id`.
- Support saving the subset of embeddings purely for that specific user. 

## Suggestions & Alternatives
- **Pre-load Model weights:** AI Models (EVA02, Faster RCNN) take large VRAM/RAM. Do NOT instantiate a new model for every upload. Load the model once globally in a worker process, and just pass the tensor data to it.
- **Cleanup Policy:** Videos are heavy. We should implement a TTL (Time To Live) cleanup cronjob (e.g., delete `/userdata/<session_id>` after 24 hours of inactivity) to save disk space.