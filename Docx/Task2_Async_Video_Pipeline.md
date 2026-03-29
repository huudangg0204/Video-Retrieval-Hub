# Task 2: Asynchronous Video Processing Pipeline (Background Worker)

## Context
Video processing (cutting frames, computing embeddings) is computationally heavy and takes minutes. Standard HTTP requests (like Flask's route) will time out (HTTP 504) if we make the user's browser wait for the entire process to finish before returning a response.
We need an asynchronous background pipeline that accepts a video, puts it in a queue, executes the steps sequentially, and updates a database/in-memory store with its physical progress so the frontend can display a loading bar.

## Requirements & Goals
1. **Background Job Execution:** Implement a mechanism to run heavy AI tasks outside the Flask request lifecycle.
2. **Video Pipeline Orchestrator:** Create a master function that strings together Task 1's modules: `Receive Video -> Extract Keyframes -> Generate Embeddings -> Build FAISS Index -> Mark Ready`.
3. **Progress Tracking Status:** Store job statuses (e.g., `PENDING`, `EXTRACTING_FRAMES`, `EMBEDDING`, `DONE`, `FAILED`) and progress percentages (0-100%).

## Action Plan (Step-by-Step)

### Step 2.1: Implement Pipeline Orchestrator
Create `workers/video_pipeline.py`. Write a function `process_video_task(session_id, raw_video_path)` that sequentially calls:
1. `extract_keyframes(raw_video_path, user_keyframe_dir)`
2. `generate_embeddings(...)`
3. `build_faiss_index(...)`

### Step 2.2: Establish Background Worker Architecture
*Option A (Lightweight - Recommended for MVP):* Use `concurrent.futures.ThreadPoolExecutor` or Python's native `threading` to run `process_video_task` in the background. Note: Flask handles threads relatively easily for simple setups.
*Option B (Robust - Recommended for Production):* Setup **Celery** + **Redis/RabbitMQ**. 

### Step 2.3: Progress State Management
- Implement a simple Dictionary/SQLite/Redis store to hold the state of `session_id`.
- Within `process_video_task`, update the state at each major milestone.
```python
# Example state dict
SESSION_STATUS = {
    "xyz-123": {"status": "EMBEDDING", "progress": 60, "message": "Extracting features using EVA02..."}
}
```
- Make sure `extract_keyframes` and `generate_embeddings` can emit progress callbacks.

## Suggestions & Alternatives
- **Progress granularity:** If `cut_keyframe` takes 30s and embedding takes 2m, map the progress properly (e.g., cut_keyframe = 0-20%, embedding = 20-95%, faiss = 95-100%).
- **Error Handling:** If the video is corrupted, the pipeline must catch the exception and update the status to `FAILED: Video corrupted`, allowing the frontend to notify the user.