import os
import time
import json
import csv
import numpy as np
from urllib.parse import unquote

from flask import Flask, request, jsonify, render_template, send_file, abort, send_from_directory
from flask_cors import CORS

# Import engine from your file
from eva02_retrieval_trake import EVA02ImageRetrieval
from workers.video_pipeline import pipeline_manager
from utils.storage import initialize_user_workspace, get_session_paths, USER_DATA_ROOT

OUTPUT_FOLDER = "submission"

# Config (nếu muốn override, hoặc đặt env vars)
PORT = int(os.environ.get("PORT", 5000))
HOST = os.environ.get("HOST", "0.0.0.0")

# Khởi tạo Flask app (templates folder expected at ./templates)
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Khởi tạo retriever
print("🔁 Initializing EVA02 retriever (this may take a while)...")
retriever = EVA02ImageRetrieval()
print("✅ Retriever ready.")

# Base dir allowed for serving images (keyframes dir from retriever)
IMG_ALLOWED_BASE = os.path.abspath(retriever.keyframes_dir) if getattr(retriever, "keyframes_dir", None) else os.path.abspath("./keyframes")

# Home page (render template)
@app.route("/")
def index():
    return render_template("index.html")

# Helper: get session retriever
def get_retriever(session_id=None):
    if not session_id:
        return retriever
    dirs = get_session_paths(session_id)
    if not dirs:
        return retriever
    
    return EVA02ImageRetrieval(
        embedding_dir=dirs["embeddings"],
        keyframes_dir=dirs["keyframes"],
        db_path=os.path.join(dirs["db"], "faiss_db.pkl"),
        map_keyframes_dir=dirs["map_keyframes"]
    )

# -----------------------
# API: Search text
# -----------------------
@app.route("/api/search_text", methods=["POST"])
def api_search_text():
    payload = request.get_json(force=True, silent=True) or {}
    query = payload.get("query", "")
    session_id = payload.get("session_id", None)
    top_k = int(payload.get("top_k", 100) or 100)
    if not query:
        return jsonify({"error": "empty query"}), 400

    # New: extract object filter parameters
    objects = payload.get("objects", "")  # comma-separated objects; if empty, hold all
    try:
        threshold = float(payload.get("threshold", 0.5))
    except ValueError:
        threshold = 0.5

    start = time.time()
    
    # Use session specific retriever if available
    local_retriever = get_retriever(session_id)
    
    results = local_retriever.search_text(
        query,
        top_k=top_k,
        save_to_db=True,
        objects=objects,
        threshold=threshold
    )
    elapsed = time.time() - start
    return jsonify({"elapsed": elapsed, "results": results})

# -----------------------
# API: Search image by existing path
# -----------------------
@app.route("/api/search_image", methods=["POST"])
def api_search_image():
    payload = request.get_json(force=True, silent=True) or {}
    img_path = payload.get("image_path", "")
    top_k = int(payload.get("top_k", 100) or 100)
    if not img_path:
        return jsonify({"error": "image_path required"}), 400
    start = time.time()
    results = retriever.search_image(img_path, top_k=top_k, save_to_db=True)
    elapsed = time.time() - start
    return jsonify({"elapsed": elapsed, "results": results})

# -----------------------
# API: Upload Video (Task 3)
# -----------------------
@app.route("/api/upload_video", methods=["POST"])
def api_upload_video():
    if "file" not in request.files:
        return jsonify({"error": "no video file"}), 400
    
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "empty filename"}), 400
        
    session_id = request.form.get("session_id")
    if session_id:
        dirs = get_session_paths(session_id)
        if not dirs:
            session_id, dirs = initialize_user_workspace()
    else:
        session_id, dirs = initialize_user_workspace()
    
    # Save video to the session's video directory
    safe_name = f.filename.replace("/", "_").replace("\\", "_")
    video_path = os.path.join(dirs["videos"], safe_name)
    f.save(video_path)
    
    # Trigger background worker
    pipeline_manager.submit_video_job(session_id, video_path, dirs)
    
    return jsonify({"session_id": session_id, "status": "processing"}), 200

# -----------------------
# API: Status Polling (Task 3)
# -----------------------
@app.route("/api/status/<session_id>", methods=["GET"])
def api_status(session_id):
    status_info = pipeline_manager.get_status(session_id)
    return jsonify(status_info), 200

# -----------------------
# API: Upload image (multipart) then search
# -----------------------
UPLOAD_DIR = os.path.abspath("./_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/api/upload_image", methods=["POST"])
def api_upload_image():
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "empty filename"}), 400
    # save
    safe_name = f.filename.replace("/", "_").replace("\\", "_")
    save_path = os.path.join(UPLOAD_DIR, f"{int(time.time()*1000)}_{safe_name}")
    f.save(save_path)
    top_k = int(request.form.get("top_k", 100) or 100)
    start = time.time()
    results = retriever.search_image(save_path, top_k=top_k, save_to_db=True)
    elapsed = time.time() - start
    return jsonify({"elapsed": elapsed, "results": results, "uploaded_path": save_path})

# (TRAKE, Export, and Submission endpoints removed based on user request)

# -----------------------
# API: Stored queries / reload
# -----------------------
@app.route("/api/stored_queries", methods=["GET"])
def api_stored_queries():
    session_id = request.args.get("session_id")
    local_retriever = get_retriever(session_id)
    return jsonify(local_retriever.get_stored_queries())

@app.route("/api/reload_query/<int:idx>", methods=["GET"])
def api_reload_query(idx):
    session_id = request.args.get("session_id")
    local_retriever = get_retriever(session_id)
    res = local_retriever.reload_query_results(idx)
    return jsonify(res)

# -----------------------
# API: get_frames_of_video
# -----------------------
@app.route("/api/get_frames/<path:video_name>", methods=["GET"])
def api_get_frames(video_name):
    session_id = request.args.get("session_id")
    local_retriever = get_retriever(session_id)
    frames = local_retriever.get_frames_of_video(video_name)
    return jsonify(frames)

# -----------------------
# Helper: Serve image files safely
# -----------------------
@app.route("/image")
def serve_image():
    # Expect query param 'path' = URL-encoded filesystem path (as retriever returns)
    raw = request.args.get("path", "")
    if not raw:
        return abort(400, "path param required")
    # decode
    decoded = unquote(raw)
    # normalize absolute
    abs_path = os.path.abspath(decoded)
    # Security: ensure abs_path is under permitted base (IMG_ALLOWED_BASE) or USER_DATA_ROOT
    try:
        common_main = os.path.commonpath([abs_path, IMG_ALLOWED_BASE])
        common_user = os.path.commonpath([abs_path, USER_DATA_ROOT])
    except ValueError:
        return abort(403)
        
    allowed_main = (common_main == IMG_ALLOWED_BASE) or abs_path.startswith(IMG_ALLOWED_BASE)
    allowed_user = (common_user == USER_DATA_ROOT) or abs_path.startswith(USER_DATA_ROOT)
    
    if not (allowed_main or allowed_user):
        # not inside allowed folder
        return abort(403)
    if not os.path.exists(abs_path):
        return abort(404)
    # send file
    return send_file(abs_path)

# -----------------------
# API: stats
# -----------------------
@app.route("/api/stats", methods=["GET"])
def api_stats():
    session_id = request.args.get("session_id")
    local_retriever = get_retriever(session_id)
    return jsonify(local_retriever.get_stats())

# -----------------------
# API: Get frames range for a video (new feature)
# -----------------------
@app.route("/api/get_frames_range/<path:video_name>/<int:frame_id>/<int:range_val>", methods=["GET"])
def api_get_frames_range(video_name, frame_id, range_val):
    session_id = request.args.get("session_id")
    local_retriever = get_retriever(session_id)
    # Get all frames for the video (this function should return a list of dicts with "image_path" key)
    frames = local_retriever.get_frames_of_video(video_name)
    # Calculate boundaries ensuring we don't exceed available frame indices.
    start_idx = max(0, frame_id - range_val)
    end_idx = min(len(frames), frame_id + range_val + 1)
    frames_range = frames[start_idx:end_idx]
    return jsonify(frames_range)

# -----------------------
# API: Search text with image
# -----------------------
@app.route("/api/search_text_with_image", methods=["POST"])
def api_search_text_with_image():
    if "file" not in request.files:
        return jsonify({"error": "no file part"}), 400
    
    # Get image file
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "empty filename"}), 400
        
    # Get other parameters
    query = request.form.get("query", "")
    top_k = int(request.form.get("top_k", 100) or 100)
    objects = request.form.get("objects", "")
    threshold = float(request.form.get("threshold", 0.5))

    if not query:
        return jsonify({"error": "empty query"}), 400

    # Save uploaded image temporarily
    safe_name = f.filename.replace("/", "_").replace("\\", "_")
    save_path = os.path.join(UPLOAD_DIR, f"{int(time.time()*1000)}_{safe_name}")
    f.save(save_path)

    start = time.time()
    
    # First get text search results
    text_results = retriever.search_text(
        query,
        top_k=top_k * 2,  # Get more results for reranking
        save_to_db=False,
        objects=objects,
        threshold=threshold
    )

    if not text_results:
        os.remove(save_path)
        return jsonify({"error": "no text results found"}), 404

    # Extract DINO-v2 features for uploaded image
    query_features = retriever.encode_image(save_path)

    # Rerank using DINO-v2 similarities
    reranked_results = []
    for candidate in text_results:
        # Get candidate image path
        candidate_path = candidate["image_path"]
        
        # Calculate DINO-v2 similarity 
        candidate_features = retriever.get_embedding(candidate_path)
        if candidate_features is not None:
            # Convert to float32 for cosine similarity
            query_features_float = query_features.astype(np.float32)
            candidate_features_float = candidate_features.astype(np.float32)
            
            # Calculate cosine similarity
            dino_similarity = float(np.dot(query_features_float[0], candidate_features_float[0]) / 
                                 (np.linalg.norm(query_features_float[0]) * np.linalg.norm(candidate_features_float[0])))

            # Combine scores (0.6 text, 0.4 DINO-v2)
            combined_score = 0.6 * candidate["similarity"] + 0.4 * dino_similarity
            
            # Update result with combined score
            candidate_copy = candidate.copy()  # Create a copy to avoid modifying original
            candidate_copy["similarity"] = float(combined_score)  # Convert to native Python float
            reranked_results.append(candidate_copy)
    
    # Sort by combined score and take top_k
    reranked_results.sort(key=lambda x: x["similarity"], reverse=True)
    final_results = reranked_results[:top_k]

    elapsed = time.time() - start
    
    # Clean up uploaded file
    try:
        os.remove(save_path)
    except:
        pass

    return jsonify({
        "elapsed": float(elapsed),  # Convert to native Python float
        "results": final_results
    })
# -----------------------
# Run server
# -----------------------
if __name__ == "__main__":
    print(f"🌐 Starting server on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=True)  # Set debug=False in production