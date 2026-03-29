import os
import time
import json
import csv
from urllib.parse import unquote

from flask import Flask, request, jsonify, render_template, send_file, abort
from flask_cors import CORS

# Import engine from your file
from eva02_retrieval_trake import EVA02ImageRetrieval

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
IMG_ALLOWED_BASE = os.path.abspath(retriever.keyframes_dir) if getattr(retriever, "keyframes_dir", None) else os.path.abspath(r"C:\NHP\work\AIC_2025\data\keyframes")

# Home page (render template)
@app.route("/")
def index():
    stats = retriever.get_stats()
    # stats may contain numpy tuple, convert to JSON-friendly
    return render_template("index.html", stats=stats)

# -----------------------
# API: Search text
# -----------------------
@app.route("/api/search_text", methods=["POST"])
def api_search_text():
    payload = request.get_json(force=True, silent=True) or {}
    query = payload.get("query", "")
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
    results = retriever.search_text(
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

# -----------------------
# API: TRAKE (Temporal Retrieval) with option (default "Closest")
# -----------------------
@app.route("/api/trake", methods=["POST"])
def api_trake():
    payload = request.get_json(force=True, silent=True) or {}
    events = payload.get("events", [])
    top_k = int(payload.get("top_k", 200) or 200)
    candidates = int(payload.get("candidates", 200) or 200)
    # Use default option "Closest" if not provided by the front end.
    option = payload.get("option", "Closest")
    if not events:
        return jsonify({"error": "events required"}), 400

    start = time.time()
    if option == "Closest":
        results = retriever.trake_closest(events, top_k=top_k, candidates= candidates)
    elif option == "Highest":
        results = retriever.trake_highest(events, top_k=top_k, candidates=candidates, option=0)
    elif option == "All":
        results = retriever.trake_highest(events, top_k=top_k, candidates=candidates, option=1)
    else:
        return jsonify({"error": "Invalid option"}), 400
    elapsed = time.time() - start
    return jsonify({"elapsed": elapsed, "result": results})

# -----------------------
# API: Stored queries / reload
# -----------------------
@app.route("/api/stored_queries", methods=["GET"])
def api_stored_queries():
    return jsonify(retriever.get_stored_queries())

@app.route("/api/reload_query/<int:idx>", methods=["GET"])
def api_reload_query(idx):
    res = retriever.reload_query_results(idx)
    return jsonify(res)

# -----------------------
# API: get_frames_of_video
# -----------------------
@app.route("/api/get_frames/<path:video_name>", methods=["GET"])
def api_get_frames(video_name):
    frames = retriever.get_frames_of_video(video_name)
    return jsonify(frames)

# -----------------------
# API: export_results
# Expect JSON:
# {
#   "results": [ { "image_path": "...", "video_name":"..", "frame_idx":"..", "similarity":.. }, ... ],
#   "name": "file_name_no_ext",
#   "const_value": optional integer or null
# }
# -----------------------
@app.route("/api/export_results", methods=["POST"])
def api_export_results():
    payload = request.get_json(force=True, silent=True) or {}
    results = payload.get("results", [])
    name = payload.get("name", f"exported_{int(time.time())}")
    const_value = payload.get("const_value", None)
    # try convert const_value to int if provided and not blank
    if const_value in ("", None):
        const_value = None
    else:
        try:
            const_value = int(const_value)
        except:
            const_value = None
    retriever.export_results(results, name, const_value)
    csv_path = os.path.join("submission", f"{name}.csv")
    return jsonify({"status": "ok", "csv_path": csv_path})

# -----------------------
# API: export_trake_results
# Expect JSON:
# {
#   "results": [ { "image_path": "...", "video_name":"..", "frame_idx":"..", "similarity":.. }, ... ],
#   "name": "file_name_no_ext",
#   "const_value": optional integer or null
# }
# -----------------------

@app.route("/api/export_trake_results", methods=["POST"])
def export_trake_results():
    payload = request.get_json(force=True, silent=True) or {}
    results = payload.get("results", [])
    name = payload.get("name", f"trake_{int(time.time())}")

    if not results:
        return jsonify({"status": "error", "msg": "No results provided"}), 400

    try:
        out_dir = os.path.join("submission")
        os.makedirs(out_dir, exist_ok=True)

        retriever.export_trake(results, name)
        csv_path = os.path.join(out_dir, f"{name}.csv")

        return jsonify({"status": "ok", "csv_path": csv_path})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500


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
    # Security: ensure abs_path is under permitted base (IMG_ALLOWED_BASE)
    try:
        common = os.path.commonpath([abs_path, IMG_ALLOWED_BASE])
    except ValueError:
        return abort(403)
    if common != IMG_ALLOWED_BASE and not abs_path.startswith(IMG_ALLOWED_BASE):
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
    return jsonify(retriever.get_stats())

# -----------------------
# API: Get frames range for a video (new feature)
# -----------------------
@app.route("/api/get_frames_range/<path:video_name>/<int:frame_id>/<int:range_val>", methods=["GET"])
def api_get_frames_range(video_name, frame_id, range_val):
    # Get all frames for the video (this function should return a list of dicts with "image_path" key)
    frames = retriever.get_frames_of_video(video_name)
    # Calculate boundaries ensuring we don't exceed available frame indices.
    start_idx = max(0, frame_id - range_val)
    end_idx = min(len(frames), frame_id + range_val + 1)
    frames_range = frames[start_idx:end_idx]
    return jsonify(frames_range)

# -----------------------
# Run server
# -----------------------
if __name__ == "__main__":
    print(f"🌐 Starting server on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=True)
