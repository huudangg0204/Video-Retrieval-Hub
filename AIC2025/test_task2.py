import os
import sys
import time

# Ensure Python can find local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.storage import initialize_user_workspace
from workers.video_pipeline import pipeline_manager

def main():
    print("=== Testing Task 2: Async Video Pipeline ===")

    # Check if a session_id was passed
    provided_session = sys.argv[1] if len(sys.argv) > 1 else None
    
    # 1. Initialize Workspace
    session_id, dirs = initialize_user_workspace(session_id=provided_session)
    
    if provided_session:
        print(f"\n[1] Resuming existing Workspace for session: {session_id}")
    else:
        print(f"\n[1] Dynamic Workspace created for session: {session_id}")
        print(f"    (To reuse this folder next time, run: python test_task2.py {session_id})")
        
    for key, path in dirs.items():
        print(f" - {key}: {path}")

    # 2. Assert Video exists
    test_video_path = os.path.join(dirs["videos"], "test_video.mp4")
    
    if not os.path.exists(test_video_path):
        print(f"\n[!] Please place a test video at {test_video_path}")
        print("Then run this test again.")
        return

    # 3. Submit Background Job
    print("\n[3] Submitting Video Job to Background Pipeline...")
    pipeline_manager.submit_video_job(session_id, test_video_path, dirs)

    # 4. Polling Loop
    print("\n[4] Polling status every 1 second (simulating API frontend requests)...")
    while True:
        status_info = pipeline_manager.get_status(session_id)
        status_enum = status_info.get("status")
        progress = status_info.get("progress", 0)
        message = status_info.get("message", "")
        
        # Display progress bar in terminal
        bar_len = 30
        filled_len = int(round(bar_len * progress / float(100)))
        bar = '=' * filled_len + '-' * (bar_len - filled_len)
        
        # Ensure we clear line correctly in terminal
        sys.stdout.write(f"\r[{status_enum}] [{bar}] {progress}% | {message}")
        sys.stdout.flush()

        if status_enum in ["DONE", "FAILED"]:
            print(f"\n\nTest Finished with Status: {status_enum}")
            break
            
        time.sleep(1.0)

if __name__ == "__main__":
    main()
