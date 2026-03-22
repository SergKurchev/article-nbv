import os
import pybullet as p
import random
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Modify path to enable running as script from anywhere
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import config
from src.simulation.camera import Camera
from src.simulation.asset_loader import AssetLoader

def generate_dataset(num_samples_per_class=100):
    client_id = p.connect(p.DIRECT)
    camera = Camera(client_id)
    loader = AssetLoader(client_id)
    
    dataset_dir = config.BASE_DIR / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    
    total_imgs = config.NUM_CLASSES * num_samples_per_class
    pbar = tqdm(total=total_imgs, desc="Generating Dataset")
    
    for class_id in range(config.NUM_CLASSES):
        class_dir = dataset_dir / f"{class_id:02d}"
        class_dir.mkdir(exist_ok=True)
        
        # Load obj
        loader.clear_obstacles()
        obj_id = loader.load_target_object(class_id)
        
        # Float it
        p.changeDynamics(obj_id, -1, mass=0)
        
        for i in range(num_samples_per_class):
            target_pos = [0.5, 0.0, 0.2] # Match environment position
            
            # 0. Generate static obstacles
            loader.clear_obstacles()
            loader.generate_obstacles()
            
            # 1. Randomize object rotation
            euler = [random.uniform(0, 2*np.pi) for _ in range(3)]
            orn = p.getQuaternionFromEuler(euler)
            p.resetBasePositionAndOrientation(obj_id, target_pos, orn, physicsClientId=client_id)
            p.stepSimulation(physicsClientId=client_id)
            
            # 2. Randomize camera position (spherical coords around target to simulate EE view)
            radius = random.uniform(0.2, 0.5) # Distance from 20cm to 50cm
            theta = random.uniform(0, 2*np.pi) # Yaw
            phi = random.uniform(0.1, np.pi/2 - 0.1) # Pitch (upper hemisphere)
            
            cam_x = target_pos[0] + radius * np.sin(phi) * np.cos(theta)
            cam_y = target_pos[1] + radius * np.sin(phi) * np.sin(theta)
            cam_z = target_pos[2] + radius * np.cos(phi)
            cam_eye = [cam_x, cam_y, cam_z]
            
            # Compute camera view matrix manually to aim at object
            camera.view_matrix = p.computeViewMatrix(
                cameraEyePosition=cam_eye,
                cameraTargetPosition=target_pos,
                cameraUpVector=[0, 0, 1], # World Z as UP
                physicsClientId=client_id
            )
            
            # Capture using updated camera matrix
            rgb, depth, seg = camera.get_image()
            
            # Save RGB
            img_path = class_dir / f"{i:04d}_rgb.png"
            Image.fromarray(rgb).save(img_path)
            
            # Save Depth (as float32 npy)
            depth_path = class_dir / f"{i:04d}_depth.npy"
            np.save(depth_path, depth)
            
            # Save Segmentation (as 8-bit png, since obj ids are small)
            # Find the object mask (where seg == obj_id)
            mask = (seg == obj_id).astype(np.uint8) * 255
            mask_path = class_dir / f"{i:04d}_mask.png"
            Image.fromarray(mask).save(mask_path)
            
            # Save Metadata: Camera Pose and Manipulator Pose (joints)
            # In dataset.py, we don't have a real robot, so we save the cam_eye as camera pose
            # and zeros for joints as placeholders (or just the cam_eye)
            metadata = {
                "cam_eye": cam_eye,
                "target_pos": target_pos,
                "class_id": class_id
            }
            import json
            meta_path = class_dir / f"{i:04d}_meta.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f)
            
            pbar.update(1)
            
        p.removeBody(obj_id, physicsClientId=client_id)
        
    p.disconnect(client_id)
    pbar.close()
    print("Dataset generation completed.")

if __name__ == "__main__":
    generate_dataset()
