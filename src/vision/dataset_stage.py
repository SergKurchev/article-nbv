"""
Stage-aware dataset generation for NBV training.

This module generates datasets for all 3 training stages:
- Stage 1: Single object at fixed position, no obstacles
- Stage 2: Multiple objects (2-10) with uniform distribution, NO obstacles
- Stage 3: Multiple objects (2-10) + obstacles (1-5) with uniform distribution
"""

import os
import pybullet as p
import pybullet_data
import random
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2
import json

# Modify path to enable running as script from anywhere
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import config
from src.simulation.camera import Camera
from src.simulation.asset_loader import AssetLoader
from src.simulation.robot import Robot


def generate_stage_dataset(num_samples_per_class=None, views_per_sample=None, use_gui=False):
    """Generate dataset using stage-aware scene generation.

    Uses AssetLoader.generate_scene() which properly handles:
    - Stage 1: Single object at fixed position
    - Stage 2: Multiple objects (2-10), NO obstacles
    - Stage 3: Multiple objects (2-10) + obstacles (1-5)

    Args:
        num_samples_per_class: Number of samples per class
        views_per_sample: Number of camera views per sample
        use_gui: Whether to use PyBullet GUI
    """
    if num_samples_per_class is None:
        num_samples_per_class = config.DATASET_SAMPLES_PER_CLASS
    if views_per_sample is None:
        views_per_sample = config.DATASET_VIEWS_PER_SAMPLE

    if use_gui:
        client_id = p.connect(p.GUI)
    else:
        client_id = p.connect(p.DIRECT)

    # Setup scene
    p.setAdditionalSearchPath(config.get_short_path(pybullet_data.getDataPath()), physicsClientId=client_id)
    plane_id = p.loadURDF("plane.urdf", physicsClientId=client_id)
    # Make ground plane black
    p.changeVisualShape(plane_id, -1, rgbaColor=[0, 0, 0, 1], physicsClientId=client_id)

    camera = Camera(client_id)
    loader = AssetLoader(client_id)

    robot_id = loader.load_robot()
    robot = Robot(client_id, robot_id)
    robot.reset()

    dataset_dir = config.DATASET_DIR
    dataset_dir.mkdir(parents=True, exist_ok=True)

    total_samples = config.NUM_CLASSES * num_samples_per_class
    pbar = tqdm(total=total_samples, desc="Generating Dataset")

    sample_counter = 0

    for class_id in range(config.NUM_CLASSES):
        i = 0
        while i < num_samples_per_class:
            # Generate scene using stage-aware method
            # Returns list of (object_id, class_id) tuples
            # - Stage 1: 1 object at fixed position
            # - Stage 2: 2-10 objects with uniform distribution
            # - Stage 3: 2-10 objects + 1-5 obstacles with uniform distribution
            objects_with_classes = loader.generate_scene()

            if len(objects_with_classes) == 0:
                print(f"Warning: No objects generated for sample {sample_counter}")
                continue
                
            sample_attempts = getattr(generate_stage_dataset, 'sample_attempts', 0)
            generate_stage_dataset.sample_attempts = sample_attempts + 1
            if generate_stage_dataset.sample_attempts > 1000:
                print(f"Error: Stuck in infinite loop generating sample {sample_counter}. Aborting class.")
                break

            # For Stage 1, we want to generate samples for each class
            # So we regenerate until we get the desired class
            if config.SCENE_STAGE == 1:
                obj_id, obj_class_id = objects_with_classes[0]
                if obj_class_id != class_id:
                    # Wrong class, regenerate
                    continue

            # Create sample directory
            sample_dir = dataset_dir / f"sample_{sample_counter:05d}"
            sample_dir.mkdir(exist_ok=True)
            (sample_dir / "rgb").mkdir(exist_ok=True)
            (sample_dir / "depth").mkdir(exist_ok=True)
            (sample_dir / "masks").mkdir(exist_ok=True)
            (sample_dir / "labels").mkdir(exist_ok=True)

            # Build color_map for all objects in scene
            color_map = []

            # Add target objects
            for obj_idx, (obj_id, obj_class_id) in enumerate(objects_with_classes):
                texture_type = ['red', 'mixed', 'green'][obj_class_id % 3]

                # For multi-object scenes, vary the red channel slightly
                # Keep it within valid uint8 range [0, 255]
                red_value = max(0, min(255, config.DATASET_COLOR_TARGET[0] - obj_idx * 10))

                color_map.append({
                    "color": [
                        red_value,
                        config.DATASET_COLOR_TARGET[1],
                        config.DATASET_COLOR_TARGET[2]
                    ],
                    "instance_id": obj_idx,
                    "category_id": obj_class_id + 1,
                    "category_name": "target_object",
                    "texture_type": texture_type
                })

            # Add robot
            color_map.append({
                "color": config.DATASET_COLOR_ROBOT,
                "instance_id": len(objects_with_classes),
                "category_id": config.NUM_CLASSES + 1,
                "category_name": "robot"
            })

            # Add obstacles (only for Stage 3)
            for obs_idx, obs_id in enumerate(loader.obstacles):
                color_map.append({
                    "color": [
                        config.DATASET_COLOR_OBSTACLE_BASE[0],
                        config.DATASET_COLOR_OBSTACLE_BASE[1],
                        config.DATASET_COLOR_OBSTACLE_BASE[2] - obs_idx * config.DATASET_COLOR_OBSTACLE_STEP
                    ],
                    "instance_id": len(objects_with_classes) + 1 + obs_idx,
                    "category_id": config.NUM_CLASSES + 2,
                    "category_name": "obstacle"
                })

            # Cache intrinsics
            intrinsics = camera.get_intrinsics()

            # Generate views with random camera positions in manipulator workspace
            cameras_data = {}
            valid_views = 0

            # Get all object and obstacle positions for collision checking
            all_positions = []
            for obj_id, _ in objects_with_classes:
                pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=client_id)
                all_positions.append(pos)
            for obs_id in loader.obstacles:
                pos, _ = p.getBasePositionAndOrientation(obs_id, physicsClientId=client_id)
                all_positions.append(pos)

            for view_idx in range(views_per_sample):
                valid_pose = False
                actual_pos = None
                actual_orn = None
                cam_target = None
                
                # Try to find valid camera position and orientation (max 200 attempts)
                for attempt in range(200):
                    # 1. Sample a target point from object generation space
                    target_x = random.uniform(config.SCENE_BOUNDS_X_MIN, config.SCENE_BOUNDS_X_MAX)
                    target_y = random.uniform(config.SCENE_BOUNDS_Y_MIN, config.SCENE_BOUNDS_Y_MAX)
                    target_z = random.uniform(config.SCENE_BOUNDS_Z_MIN, config.SCENE_BOUNDS_Z_MAX)
                    cam_target = [target_x, target_y, target_z]

                    # 2. Sample cam_eye within manipulator reach (radius 0.8) and above floor
                    cam_x = random.uniform(-0.8, 0.8)
                    cam_y = random.uniform(-0.8, 0.8)
                    cam_z = random.uniform(0.05, 0.8)
                    candidate_pos = [cam_x, cam_y, cam_z]

                    if np.linalg.norm(candidate_pos) > 0.8:
                        continue # Outside physical reach of robot

                    # Basic check to avoid spawning exactly inside objects
                    too_close = False
                    for obj_pos in all_positions:
                        if np.linalg.norm(np.array(candidate_pos) - np.array(obj_pos)) < 0.1:
                            too_close = True
                            break
                    if too_close:
                        continue

                    # Compute direction to target
                    diff = np.array(cam_target) - np.array(candidate_pos)
                    norm = np.linalg.norm(diff)
                    if norm < 0.1:
                        continue
                        
                    # Calculate pitch and yaw to look at target
                    import math
                    yaw = math.atan2(diff[1], diff[0])
                    pitch = math.asin(np.clip(diff[2] / norm, -1.0, 1.0))
                    
                    # To align Kuka's +Z axis with the target direction,
                    # and +X axis as UP vector:
                    # Apply the correct pitch rotation (pi/2 - pitch)
                    cam_orn = p.getQuaternionFromEuler([0, math.pi/2 - pitch, yaw])

                    # Apply action to move robot
                    robot.apply_action(candidate_pos, cam_orn)
                    p.stepSimulation(physicsClientId=client_id)

                    # 3. Check for collisions of the robot
                    collision = False
                    # Check collision with obstacles
                    for obs_id in loader.obstacles:
                        pts = p.getClosestPoints(bodyA=robot.robot_id, bodyB=obs_id, distance=0.01, physicsClientId=client_id)
                        if pts:
                            collision = True
                            break
                            
                    if not collision:
                        # Check collision with objects
                        for obj_id, _ in objects_with_classes:
                            pts = p.getClosestPoints(bodyA=robot.robot_id, bodyB=obj_id, distance=0.01, physicsClientId=client_id)
                            if pts:
                                collision = True
                                break
                                
                    if not collision:
                        # Check collision with floor
                        pts = p.getClosestPoints(bodyA=robot.robot_id, bodyB=plane_id, distance=0.01, physicsClientId=client_id)
                        for pt in pts:
                            if pt[3] > 2: # Ignore base links which might touch the floor naturally
                                collision = True
                                break

                    if collision:
                        continue

                    # 4. Check reachability (did IK actually reach the candidate pos?)
                    actual_pos, actual_orn = robot.get_ee_pose()
                    if np.linalg.norm(np.array(actual_pos) - np.array(candidate_pos)) > 0.1:
                        continue # Unreachable

                    # 5. Capture image and check visibility
                    rgb, depth, seg = camera.get_image(cam_pos=actual_pos, cam_orn=actual_orn)
                    
                    # Check if any target object is actually visible (at least N pixels)
                    target_visible = False
                    for obj_id_check, _ in objects_with_classes:
                        if np.sum(seg == obj_id_check) >= config.DATASET_MIN_OBJECT_PIXELS:
                            target_visible = True
                            break
                    
                    if not target_visible:
                        continue # Try another pose for this view_idx

                    valid_pose = True
                    break

                if not valid_pose:
                    # Could not find valid collision-free reachable camera position WITH target visible
                    continue

                # Save RGB and data since we now have a valid visible pose
                Image.fromarray(rgb).save(sample_dir / "rgb" / f"{view_idx:05d}.png")

                # Save depth
                np.save(sample_dir / "depth" / f"{view_idx:05d}.npy", depth)

                # Create instance mask with colors from color_map
                h, w = seg.shape
                mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)

                # Target objects
                for obj_idx, (obj_id, _) in enumerate(objects_with_classes):
                    red_value = max(0, min(255, config.DATASET_COLOR_TARGET[0] - obj_idx * 10))
                    mask_rgb[seg == obj_id] = [
                        red_value,
                        config.DATASET_COLOR_TARGET[1],
                        config.DATASET_COLOR_TARGET[2]
                    ]

                # Robot
                mask_rgb[seg == robot_id] = config.DATASET_COLOR_ROBOT

                # Obstacles
                for obs_idx, obs_id in enumerate(loader.obstacles):
                    mask_rgb[seg == obs_id] = [
                        config.DATASET_COLOR_OBSTACLE_BASE[0],
                        config.DATASET_COLOR_OBSTACLE_BASE[1],
                        config.DATASET_COLOR_OBSTACLE_BASE[2] - obs_idx * config.DATASET_COLOR_OBSTACLE_STEP
                    ]

                Image.fromarray(mask_rgb).save(sample_dir / "masks" / f"{view_idx:05d}.png")

                # YOLO labels for all target objects
                yolo_lines = []
                for obj_idx, (obj_id, obj_class_id) in enumerate(objects_with_classes):
                    obj_mask = (seg == obj_id).astype(np.uint8)
                    contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for cnt in contours:
                        if len(cnt) < 3:
                            continue
                        x, y, w_box, h_box = cv2.boundingRect(cnt)
                        x_center = (x + w_box / 2) / config.IMAGE_SIZE
                        y_center = (y + h_box / 2) / config.IMAGE_SIZE
                        w_norm = w_box / config.IMAGE_SIZE
                        h_norm = h_box / config.IMAGE_SIZE
                        yolo_lines.append(f"{obj_class_id + 1} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

                with open(sample_dir / "labels" / f"{view_idx:05d}.txt", "w") as f:
                    f.write("\n".join(yolo_lines))

                # Store camera data
                rotation = camera.get_rotation_quaternion()
                actual_up = camera.get_up_vector(cam_orn=actual_orn)
                cam_actual_pos = camera.get_camera_position()
                
                 # Use actual_pos for robot pose, but cam_actual_pos for the camera
                cameras_data[f"{view_idx:05d}"] = {
                    "position": [float(x) for x in cam_actual_pos],
                    "target": [float(x) for x in cam_target],
                    "up": [float(x) for x in actual_up],
                    "rotation": [float(x) for x in rotation],
                    "intrinsics": intrinsics,
                    "joint_states": [float(x) for x in robot.get_joint_states()],
                    "ee_position": [float(x) for x in actual_pos],
                    "ee_quaternion": [float(x) for x in actual_orn]
                }
                valid_views += 1

            # Only save sample if we got enough valid views
            if valid_views >= config.DATASET_MIN_VALID_VIEWS:
                # Save metadata
                with open(sample_dir / "color_map.json", "w") as f:
                    json.dump(color_map, f, indent=2)
                with open(sample_dir / "cameras.json", "w") as f:
                    json.dump(cameras_data, f, indent=2)

                sample_counter += 1
                i += 1
                generate_stage_dataset.sample_attempts = 0
                pbar.update(1)
            else:
                # Remove incomplete sample
                import shutil
                shutil.rmtree(sample_dir)

    p.disconnect(client_id)
    pbar.close()
    print(f"Dataset generation completed. Created {sample_counter} samples.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    generate_stage_dataset(use_gui=args.gui)
