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


def generate_stage_dataset(num_samples_per_class=None, views_per_sample=None):
    """Generate dataset using stage-aware scene generation.

    Uses AssetLoader.generate_scene() which properly handles:
    - Stage 1: Single object at fixed position
    - Stage 2: Multiple objects (2-10), NO obstacles
    - Stage 3: Multiple objects (2-10) + obstacles (1-5)

    Args:
        num_samples_per_class: Number of samples per class
        views_per_sample: Number of camera views per sample
    """
    if num_samples_per_class is None:
        num_samples_per_class = config.DATASET_SAMPLES_PER_CLASS
    if views_per_sample is None:
        views_per_sample = config.DATASET_VIEWS_PER_SAMPLE

    client_id = p.connect(p.DIRECT)

    # Setup scene
    p.setAdditionalSearchPath(config.get_short_path(pybullet_data.getDataPath()), physicsClientId=client_id)
    p.loadURDF("plane.urdf", physicsClientId=client_id)

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

            # Compute scene center for camera orbit
            if config.SCENE_STAGE == 1:
                # Stage 1: orbit around fixed position
                scene_center = config.DATASET_TARGET_POS
            else:
                # Stage 2/3: orbit around center of all objects
                positions = []
                for obj_id, _ in objects_with_classes:
                    pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=client_id)
                    positions.append(pos)
                scene_center = np.mean(positions, axis=0).tolist()

            # Generate views around the scene
            cameras_data = {}
            valid_views = 0

            for view_idx in range(views_per_sample):
                # Spherical coordinates around scene center
                radius = random.uniform(config.DATASET_CAMERA_RADIUS_MIN, config.DATASET_CAMERA_RADIUS_MAX)
                theta = (view_idx / views_per_sample) * 2 * np.pi + random.uniform(-config.DATASET_CAMERA_THETA_JITTER, config.DATASET_CAMERA_THETA_JITTER)
                phi = random.uniform(config.DATASET_CAMERA_PHI_MIN, config.DATASET_CAMERA_PHI_MAX)

                cam_x = scene_center[0] + radius * np.sin(phi) * np.cos(theta)
                cam_y = scene_center[1] + radius * np.sin(phi) * np.sin(theta)
                cam_z = scene_center[2] + radius * np.cos(phi)
                cam_eye = [cam_x, cam_y, cam_z]

                # Compute view matrix
                camera.view_matrix = p.computeViewMatrix(
                    cameraEyePosition=cam_eye,
                    cameraTargetPosition=scene_center,
                    cameraUpVector=[0, 0, 1],
                    physicsClientId=client_id
                )

                # Capture image
                rgb, depth, seg = camera.get_image()

                # Check visibility of at least one target object
                total_obj_pixels = 0
                for obj_id, _ in objects_with_classes:
                    total_obj_pixels += np.sum(seg == obj_id)

                if total_obj_pixels < config.DATASET_MIN_OBJECT_PIXELS:
                    continue

                # Save RGB
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
                cameras_data[f"{view_idx:05d}"] = {
                    "position": cam_eye,
                    "target": scene_center,
                    "up": [0, 0, 1],
                    "rotation": rotation,
                    "intrinsics": intrinsics
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
                pbar.update(1)
            else:
                # Remove incomplete sample
                import shutil
                shutil.rmtree(sample_dir)

    p.disconnect(client_id)
    pbar.close()
    print(f"Dataset generation completed. Created {sample_counter} samples.")


if __name__ == "__main__":
    generate_stage_dataset()
