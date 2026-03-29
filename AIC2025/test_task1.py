import os
import sys

# Ensure Python can find utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.storage import initialize_user_workspace
from cut_keyframe import extract_keyframes
from utils.embedding_utils import generate_session_embeddings
from eva02_retrieval_trake import EVA02ImageRetrieval

def main():
    print("=== Testing Task 1: Module Refactoring ===")
    
    # Check if a session_id was passed via arguments
    provided_session = sys.argv[1] if len(sys.argv) > 1 else None

    # 1. Test Workspace Initialization
    session_id, dirs = initialize_user_workspace(session_id=provided_session)
    if provided_session:
        print(f"\n[1] Resuming existing Workspace for session: {session_id}")
    else:
        print(f"\n[1] Dynamic Workspace created for session: {session_id}")
        print(f"    (To reuse this folder next time, run: python test_task1.py {session_id})")
    print(f"Paths initialized:")
    for key, path in dirs.items():
        print(f" - {key}: {path}")
        assert os.path.exists(path), f"Path {path} does not exist!"

    # 2. Test Keyframe Extraction
    # Note: You need a real test video for this to work. Please replace 'test.mp4' with an actual video path if you have one.
    test_video_path = os.path.join(dirs["videos"], "test_video.mp4")
    
    # Let's create a dummy video if it doesn't exist just to show the path logic
    # (cv2 will fail to open a fake video, but the paths are what we want to test)
    if not os.path.exists(test_video_path):
        print(f"\n[!] Please place a test video at {test_video_path} to fully test extraction.")
        print(f"For now, we will simulate the extraction function call.")
    else:
        print("\n[2] Extracting Keyframes...")
        num_frames = extract_keyframes(test_video_path, dirs["root"], threshold=0.95)
        print(f"Extracted {num_frames} frames.")

    # 3. Test Embedding generation
    print("\n[3] Generating Embeddings (Loading EVA02 Singleton)...")
    try:
        # This will load EVA02 into memory 
        generate_session_embeddings(dirs["root"])
        print("Embedding generation completed without errors.")
    except Exception as e:
        print(f"Embedding generation failed: {e}")

    # 4. Test Dynamic FAISS Search and DB Logic
    print("\n[4] Initializing Session-specific Retrieval Engine...")
    try:
        # Load the retrieval engine using the dynamic paths
        retrieval = EVA02ImageRetrieval(
            embedding_dir=dirs["embeddings"],
            keyframes_dir=dirs["keyframes"],
            db_path=os.path.join(dirs["db"], "faiss_db.pkl"),
            map_keyframes_dir=dirs["map_keyframes"]
        )
        print("Retrieval Engine initialized successfully.")
        
        # Test a dummy text search to see if it saves to the db
        print("Testing a dummy query 'a person walking'...")
        results = retrieval.search_text("a person walking", top_k=5, save_to_db=True)
        print(f"Search returned {len(results)} results.")
        
        db_file = os.path.join(dirs["db"], "faiss_db.pkl")
        if os.path.exists(db_file):
            print(f"FAISS database successfully saved at: {db_file}")
        else:
            print("FAISS database not saved (might be empty embeddings).")
            
    except Exception as e:
        print(f"Retrieval Engine test failed: {e}")

    print("\n=== Test complete. Please check the 'userdata' folder! ===")

if __name__ == "__main__":
    main()
