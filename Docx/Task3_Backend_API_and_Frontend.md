# Task 3: Backend API Endpoints & Dynamic Web UI

## Context
With the heavy AI core parameterized (Task 1) and wrapped in a background worker (Task 2), the final step is exposing this architecture to the user interface. We need to modify the Flask Server (`sever.py`) endpoints and the Frontend (`index.html`) to facilitate video uploading, progress tracking, and session-based searching.

## Requirements & Goals
1. **Upload API:** An endpoint to receive `.mp4` files, generate a `session_id`, and trigger the background pipeline.
2. **Polling API:** An endpoint for the frontend to query the real-time status of their video processing.
3. **Session-based Search API:** Update the text search API to target a specific `session_id`'s FAISS index.
4. **Frontend Overhaul:** Add a drag-and-drop video upload zone, a progress bar UI, and toggle the search bar only after processing is complete.

## Action Plan (Step-by-Step)

### Step 3.1: Develop Upload API (`/api/upload_video`)
- Accept `multipart/form-data` containing a video file.
- Generate a UUID for `session_id`.
- Save the video to `/userdata/<session_id>/videos/`.
- Trigger the background worker `process_video_task(session_id)`.
- Return `{"session_id": "<uuid>", "status": "processing"}` to the frontend immediately.

### Step 3.2: Develop Status Polling API (`/api/status/<session_id>`)
- Check the store (Dictionary/Redis) for the `session_id`.
- Return JSON containing `state`, `progress_percent`, and `error_message` (if any).

### Step 3.3: Update Retrieval Endpoints (`/api/search_text`)
- Require `session_id` in the JSON payload.
- In the route, initialize/retrieve the `EVA02ImageRetrieval` instance tailored to that `session_id`.
- Perform the search and return local paths. *Ensure the image serving route (`/image`) is securely configured to read from dynamic `/userdata` paths.*

### Step 3.4: Overhaul Frontend (HTML/JS)
- **Upload UI:** Create an upload input (hidden/visible toggle based on state).
- **Progress Bar:** Use `setInterval` in Javascript to poll `/api/status/<session_id>` every 2 seconds. Update UI progress bar and text (e.g., "50% - Generating Embeddings...").
- **Search UI Transition:** Once the polling API returns `status: "DONE"`, hide the progress bar, display the Text Search input box, and append the `session_id` to all future fetch requests to `/api/search_text`.

## Suggestions & Alternatives
- **WebSockets:** Instead of short polling (HTTP intervals), you could use `Flask-SocketIO` to push progress updates directly to the client. This is cleaner and more real-time.
- **Session Persistence:** Store the `session_id` in `localStorage` or browser cookies. If the user refreshes the page, they don't have to upload the video again (just reload the search interface).