import pybullet as p
import pybullet_data
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src.simulation.camera import Camera
from src.simulation.asset_loader import AssetLoader

def debug_capture():
    client_id = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(config.get_short_path(pybullet_data.getDataPath()))
    plane_id = p.loadURDF("plane.urdf")
    
    loader = AssetLoader(client_id)
    # Generate a simple scene (Stage 1)
    config.SCENE_STAGE = 1
    objects = loader.generate_scene()
    obj_id, _ = objects[0]
    
    camera = Camera(client_id)
    # Move camera to look at the object
    # Default object pos is [0.5, 0.0, 0.2]
    eye_pos = [1.0, 0.0, 0.5]
    target_pos = [0.5, 0.0, 0.2]
    up_vector = [0, 0, 1]
    
    view_matrix = p.computeViewMatrix(eye_pos, target_pos, up_vector)
    
    # Capture with both renderers
    for renderer_name, renderer in [("OPENGL", p.ER_BULLET_HARDWARE_OPENGL), ("TINY", p.ER_TINY_RENDERER)]:
        w, h, rgb, depth, seg = p.getCameraImage(
            224, 224, view_matrix, camera.projection_matrix, renderer=renderer
        )
        seg_arr = np.reshape(seg, (h, w))
        unique_ids = np.unique(seg_arr)
        print(f"Renderer: {renderer_name}")
        print(f"  Unique IDs in seg: {unique_ids}")
        print(f"  Plane ID: {plane_id}")
        print(f"  Object ID: {obj_id}")
        
        # Check if object is in seg
        if obj_id in unique_ids:
            print(f"  [OK] Object {obj_id} found in segmentation!")
        else:
            print(f"  [ERROR] Object {obj_id} NOT found in segmentation!")
            
    p.disconnect()

if __name__ == "__main__":
    debug_capture()
