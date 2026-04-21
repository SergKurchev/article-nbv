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

def mask_to_polygons(mask):
    """Convert binary mask to COCO-style polygons."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if len(cnt) < 3: continue
        polygons.append(cnt.reshape(-1).tolist())
    return polygons

def mask_to_yolo(mask, class_id, img_w, img_h):
    """Convert binary mask to YOLO segmentation format string."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        if len(cnt) < 3: continue
        poly = cnt.astype(float).reshape(-1, 2)
        poly[:, 0] /= img_w
        poly[:, 1] /= img_h
        line = f"{class_id} " + " ".join([f"{c:.6f}" for c in poly.reshape(-1)])
        lines.append(line)
    return lines

def generate_dataset(num_samples_per_class=None, views_per_sample=None):
    """Generate dataset with proper structure: sample_NNNNN/ with multiple views per object.

    Args:
        num_samples_per_class: Number of different object instances per class
        views_per_sample: Number of camera views around each object
    """
    if num_samples_per_class is None:
        num_samples_per_class = config.DATASET_SAMPLES_PER_CLASS
    if views_per_sample is None:
        views_per_sample = config.DATASET_VIEWS_PER_SAMPLE

    client_id = p.connect(p.DIRECT)

    # --- Spawn Full Scene Elements ---
    p.setAdditionalSearchPath(config.get_short_path(pybullet_data.getDataPath()), physicsClientId=client_id)
    p.loadURDF("plane.urdf", physicsClientId=client_id)

    camera = Camera(client_id)
    loader = AssetLoader(client_id)

    robot_id = loader.load_robot()
    robot = Robot(client_id, robot_id)
    robot.reset()
    # ---------------------------------

    dataset_dir = config.DATASET_DIR
    dataset_dir.mkdir(parents=True, exist_ok=True)

    total_samples = config.NUM_CLASSES * num_samples_per_class
    pbar = tqdm(total=total_samples, desc="Generating Dataset")

    sample_counter = 0

    for class_id in range(config.NUM_CLASSES):
        # Determine texture type based on class_id (3-class system)
        texture_type = ['red', 'mixed', 'green'][class_id % 3]

        # Load obj with appropriate texture
        loader.clear_obstacles()
        obj_id = loader.load_target_object(class_id, texture_type=texture_type)

        # Float it
        p.changeDynamics(obj_id, -1, mass=0)

        i = 0
        while i < num_samples_per_class:
            target_pos = config.DATASET_TARGET_POS

            # 1. Randomize object and obstacles until no collision
            success_placement = False
            for _ in range(config.DATASET_PLACEMENT_ATTEMPTS):
                loader.clear_obstacles()
                loader.generate_obstacles()

                # Randomize object rotation
                euler = [random.uniform(0, 2*np.pi) for _ in range(3)]
                orn = p.getQuaternionFromEuler(euler)
                p.resetBasePositionAndOrientation(obj_id, target_pos, orn, physicsClientId=client_id)

                # Check collision between obj and ANY obstacle or robot
                collision = False
                for other_id in loader.obstacles + [robot_id]:
                    pts = p.getClosestPoints(obj_id, other_id, distance=config.DATASET_COLLISION_MARGIN, physicsClientId=client_id)
                    if len(pts) > 0:
                        collision = True
                        break

                if not collision:
                    success_placement = True
                    break

            if not success_placement:
                continue

            # Create sample directory
            sample_dir = dataset_dir / f"sample_{sample_counter:05d}"
            sample_dir.mkdir(exist_ok=True)
            (sample_dir / "rgb").mkdir(exist_ok=True)
            (sample_dir / "depth").mkdir(exist_ok=True)
            (sample_dir / "masks").mkdir(exist_ok=True)
            (sample_dir / "labels").mkdir(exist_ok=True)

            # Pre-compute color_map (static, same for all views)
            color_map = [
                {
                    "color": config.DATASET_COLOR_TARGET,
                    "instance_id": 0,
                    "category_id": class_id + 1,
                    "category_name": "target_object",
                    "texture_type": texture_type
                },
                {"color": config.DATASET_COLOR_ROBOT, "instance_id": 1, "category_id": config.NUM_CLASSES + 1, "category_name": "robot"}
            ]
            for obs_idx in range(len(loader.obstacles)):
                color_map.append({
                    "color": [config.DATASET_COLOR_OBSTACLE_BASE[0],
                             config.DATASET_COLOR_OBSTACLE_BASE[1],
                             config.DATASET_COLOR_OBSTACLE_BASE[2] - obs_idx * config.DATASET_COLOR_OBSTACLE_STEP],
                    "instance_id": 2 + obs_idx,
                    "category_id": config.NUM_CLASSES + 2,
                    "category_name": "obstacle"
                })

            # Cache intrinsics (same for all views)
            intrinsics = camera.get_intrinsics()

            # Generate views around the object
            cameras_data = {}
            valid_views = 0

            for view_idx in range(views_per_sample):
                # Spherical coordinates around target
                radius = random.uniform(config.DATASET_CAMERA_RADIUS_MIN, config.DATASET_CAMERA_RADIUS_MAX)
                theta = (view_idx / views_per_sample) * 2 * np.pi + random.uniform(-config.DATASET_CAMERA_THETA_JITTER, config.DATASET_CAMERA_THETA_JITTER)
                phi = random.uniform(config.DATASET_CAMERA_PHI_MIN, config.DATASET_CAMERA_PHI_MAX)

                cam_x = target_pos[0] + radius * np.sin(phi) * np.cos(theta)
                cam_y = target_pos[1] + radius * np.sin(phi) * np.sin(theta)
                cam_z = target_pos[2] + radius * np.cos(phi)
                cam_eye = [cam_x, cam_y, cam_z]

                # Compute view matrix
                camera.view_matrix = p.computeViewMatrix(
                    cameraEyePosition=cam_eye,
                    cameraTargetPosition=target_pos,
                    cameraUpVector=[0, 0, 1],
                    physicsClientId=client_id
                )

                # Capture image
                rgb, depth, seg = camera.get_image()

                # Check visibility
                obj_pixels = np.sum(seg == obj_id)
                if obj_pixels < config.DATASET_MIN_OBJECT_PIXELS:
                    continue

                # Save RGB
                Image.fromarray(rgb).save(sample_dir / "rgb" / f"{view_idx:05d}.png")

                # Save depth
                np.save(sample_dir / "depth" / f"{view_idx:05d}.npy", depth)

                # Create instance mask with colors from color_map
                h, w = seg.shape
                mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)

                # Target object
                mask_rgb[seg == obj_id] = config.DATASET_COLOR_TARGET
                # Robot
                mask_rgb[seg == robot_id] = config.DATASET_COLOR_ROBOT
                # Obstacles
                for obs_idx, obs_id in enumerate(loader.obstacles):
                    mask_rgb[seg == obs_id] = [config.DATASET_COLOR_OBSTACLE_BASE[0],
                                                config.DATASET_COLOR_OBSTACLE_BASE[1],
                                                config.DATASET_COLOR_OBSTACLE_BASE[2] - obs_idx * config.DATASET_COLOR_OBSTACLE_STEP]

                Image.fromarray(mask_rgb).save(sample_dir / "masks" / f"{view_idx:05d}.png")

                # YOLO labels (bounding box format)
                obj_mask = (seg == obj_id).astype(np.uint8)
                contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                yolo_lines = []
                for cnt in contours:
                    if len(cnt) < 3:
                        continue
                    x, y, w_box, h_box = cv2.boundingRect(cnt)
                    x_center = (x + w_box / 2) / config.IMAGE_SIZE
                    y_center = (y + h_box / 2) / config.IMAGE_SIZE
                    w_norm = w_box / config.IMAGE_SIZE
                    h_norm = h_box / config.IMAGE_SIZE
                    yolo_lines.append(f"{class_id + 1} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

                with open(sample_dir / "labels" / f"{view_idx:05d}.txt", "w") as f:
                    f.write("\n".join(yolo_lines))

                # Store camera data (rotation computed once per view)
                rotation = camera.get_rotation_quaternion()
                cameras_data[f"{view_idx:05d}"] = {
                    "position": cam_eye,
                    "rotation": rotation,
                    "intrinsics": intrinsics
                }

                valid_views += 1

            # Only save sample if we got enough valid views
            if valid_views >= config.DATASET_MIN_VALID_VIEWS:
                # Save color_map.json and cameras.json once at the end
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

        p.removeBody(obj_id, physicsClientId=client_id)

    p.disconnect(client_id)
    pbar.close()
    print(f"Dataset generation completed. Created {sample_counter} samples.")

if __name__ == "__main__":
    generate_dataset()
