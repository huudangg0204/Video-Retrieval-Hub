import os
import uuid
import shutil

# Make sure this points to a 'userdata' directory relative to the project root
# Using '..' assumes this file is in 'AI-Challenge/AIC2025/utils'
USER_DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'userdata'))

def initialize_user_workspace(session_id=None):
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    base_dir = os.path.join(USER_DATA_ROOT, session_id)
    
    dirs = {
        'root': base_dir,
        'videos': os.path.join(base_dir, 'videos'),
        'keyframes': os.path.join(base_dir, 'keyframes'),
        'map_keyframes': os.path.join(base_dir, 'map_keyframes'),
        'embeddings': os.path.join(base_dir, 'embeddings'),
        'db': os.path.join(base_dir, 'db')
    }
    
    for key, path in dirs.items():
        if key != 'root':
            os.makedirs(path, exist_ok=True)
            
    return session_id, dirs

def cleanup_workspace(session_id):
    path = os.path.join(USER_DATA_ROOT, session_id)
    if os.path.exists(path):
        shutil.rmtree(path)
        return True
    return False

def get_session_paths(session_id):
    base_dir = os.path.join(USER_DATA_ROOT, session_id)
    if not os.path.exists(base_dir):
        return None
        
    return {
        'root': base_dir,
        'videos': os.path.join(base_dir, 'videos'),
        'keyframes': os.path.join(base_dir, 'keyframes'),
        'map_keyframes': os.path.join(base_dir, 'map_keyframes'),
        'embeddings': os.path.join(base_dir, 'embeddings'),
        'db': os.path.join(base_dir, 'db')
    }
