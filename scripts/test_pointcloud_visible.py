import sys
import os
import json
import shutil
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

import pybullet as p
import pybullet_data
import time
import math
import numpy as np

import config
from src.simulation.asset_loader import AssetLoader
from src.simulation.camera import Camera

def main():
    client_id = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client_id)
    p.setAdditionalSearchPath(config.get_short_path(pybullet_data.getDataPath()), physicsClientId=client_id)
    p.loadURDF("plane.urdf", physicsClientId=client_id)
    
    asset_loader = AssetLoader(client_id)
    
    # Load 8 objects in a circle
    radius = 0.3
    center = [0.5, 0.0, 0.15]
    for i in range(8):
        angle = i * (2 * math.pi / 8)
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        z = center[2]
        asset_loader._create_object_at_position(i, 'red', [x, y, z])
    
    # Add one obstacle in the middle (gray slab)
    half_extents = [0.02, 0.15, 0.2]
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.5, 0.5, 0.5, 1], physicsClientId=client_id)
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=client_id)
    orn = p.getQuaternionFromEuler([0, 0, math.pi/4])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape, basePosition=center, baseOrientation=orn, physicsClientId=client_id)
    
    cam = Camera(client_id)
    
    # Setup test sample directory
    sample_dir = config.BASE_DIR / "dataset" / "primitives" / "test_sample"
    if sample_dir.exists():
        shutil.rmtree(sample_dir)
    
    (sample_dir / "rgb").mkdir(parents=True)
    (sample_dir / "depth").mkdir(parents=True)
    (sample_dir / "masks").mkdir(parents=True)
    
    cameras_dict = {}
    
    # 5 frames
    for i in range(5):
        angle = i * (2 * math.pi / 5)
        dist = 1.0
        cam_pos = [
            center[0] + dist * math.cos(angle),
            center[1] + dist * math.sin(angle),
            0.6
        ]
        
        diff = np.array(center) - np.array(cam_pos)
        diff = diff / np.linalg.norm(diff)
        
        # We need a proper PyBullet view matrix looking at center, with Z up
        cam_target = center
        cam_up = [0, 0, 1]
        
        cam.view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=cam_target,
            cameraUpVector=cam_up,
            physicsClientId=client_id
        )
        
        rgb, depth, seg = cam.get_image()
        
        # Save images
        view_str = f"{i:04d}"
        Image.fromarray(rgb).save(sample_dir / "rgb" / f"{view_str}.png")
        np.save(sample_dir / "depth" / f"{view_str}.npy", depth)
        
        # Dummy mask
        mask = np.zeros_like(rgb)
        Image.fromarray(mask).save(sample_dir / "masks" / f"{view_str}.png")
        
        cameras_dict[view_str] = {
            "intrinsics": cam.get_intrinsics(),
            "position": cam_pos,
            "target": cam_target,
            "up": cam_up,
            "rotation": [0,0,0,1] # dummy rotation, not used by unproject_frame when target and up are available
        }
        
    p.disconnect(client_id)
    
    with open(sample_dir / "cameras.json", "w") as f:
        json.dump(cameras_dict, f, indent=4)
        
    # Dummy color map
    color_map = [
        {"instance_id": 1, "category_id": 1, "category_name": "target_object", "color": [255, 0, 0]}
    ]
    with open(sample_dir / "color_map.json", "w") as f:
        json.dump(color_map, f, indent=4)
        
    print(f"Sample generated at {sample_dir}")
    print("Now running HTML viewer generation...")
    
    os.system(f"uv run python scripts/generate_sample_viewer.py {sample_dir}")

if __name__ == "__main__":
    main()
