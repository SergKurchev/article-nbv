"""PointNet dataset generator.

Converts RGB-D images from the existing dataset into point clouds for PointNet training.

Usage:
    uv run python src/vision/dataset_pointnet.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import json
import config
from tqdm import tqdm
import cv2


def depth_to_point_cloud(depth, intrinsics, max_depth=2.0):
    """Convert depth map to 3D point cloud.

    Args:
        depth: [H, W] depth map in meters
        intrinsics: Camera intrinsics dict with fx, fy, cx, cy
        max_depth: Maximum depth to consider (meters)

    Returns:
        points: [N, 3] point cloud (x, y, z)
    """
    h, w = depth.shape
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']

    # Create pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Filter valid depth values
    valid_mask = (depth > 0) & (depth < max_depth)

    # Get valid pixels
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    z_valid = depth[valid_mask]

    # Back-project to 3D
    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy
    z = z_valid

    points = np.stack([x, y, z], axis=1)

    return points


def sample_point_cloud(points, num_points=1024):
    """Sample fixed number of points from point cloud.

    Args:
        points: [N, 3] point cloud
        num_points: Target number of points

    Returns:
        sampled_points: [num_points, 3] sampled point cloud
    """
    n = points.shape[0]

    if n == 0:
        # Return zeros if no points
        return np.zeros((num_points, 3), dtype=np.float32)

    if n >= num_points:
        # Random sampling
        indices = np.random.choice(n, num_points, replace=False)
    else:
        # Oversample with replacement
        indices = np.random.choice(n, num_points, replace=True)

    return points[indices]


def normalize_point_cloud(points):
    """Normalize point cloud to unit sphere.

    Args:
        points: [N, 3] point cloud

    Returns:
        normalized_points: [N, 3] normalized point cloud
    """
    # Center at origin
    centroid = np.mean(points, axis=0)
    points = points - centroid

    # Scale to unit sphere
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist

    return points


def generate_pointnet_dataset():
    """Generate PointNet dataset from existing RGB-D dataset."""
    print("="*60)
    print("PointNet Dataset Generation")
    print("="*60)

    # Check if source dataset exists
    if not config.DATASET_DIR.exists():
        print(f"Error: Dataset directory not found: {config.DATASET_DIR}")
        print("Please run 'uv run python src/vision/dataset.py' first to generate RGB-D dataset.")
        return

    # Create output directory
    output_dir = config.BASE_DIR / "dataset_pointnet" / config.OBJECT_MODE
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSource dataset: {config.DATASET_DIR}")
    print(f"Output directory: {output_dir}")
    print(f"Number of points per sample: {config.POINTNET_NUM_POINTS}")

    # Get all samples
    samples = sorted([d for d in config.DATASET_DIR.iterdir() if d.is_dir() and d.name.startswith("sample_")])

    if len(samples) == 0:
        print(f"Error: No samples found in {config.DATASET_DIR}")
        return

    print(f"Found {len(samples)} samples")

    # Process each sample
    processed_count = 0
    skipped_count = 0

    for sample_dir in tqdm(samples, desc="Converting to point clouds"):
        sample_name = sample_dir.name

        # Load cameras.json
        cameras_path = sample_dir / "cameras.json"
        if not cameras_path.exists():
            skipped_count += 1
            continue

        with open(cameras_path, 'r') as f:
            cameras = json.load(f)

        # Load color_map.json to get class info
        color_map_path = sample_dir / "color_map.json"
        if not color_map_path.exists():
            skipped_count += 1
            continue

        with open(color_map_path, 'r') as f:
            color_map = json.load(f)

        # Find target object class
        target_class = None
        for entry in color_map:
            if entry['category_name'] == 'target_object':
                target_class = entry['category_id']
                break

        if target_class is None:
            skipped_count += 1
            continue

        # Convert class_id to 0-indexed (category_id is 1-indexed)
        target_class = target_class - 1

        # Process each view
        view_ids = sorted(cameras.keys())
        point_clouds = []
        vectors = []

        for view_id in view_ids:
            # Load depth map
            depth_path = sample_dir / "depth" / f"{view_id}.npy"
            if not depth_path.exists():
                continue

            depth = np.load(depth_path)

            # Get camera intrinsics
            intrinsics = cameras[view_id]['intrinsics']

            # Convert to point cloud
            points = depth_to_point_cloud(depth, intrinsics)

            if points.shape[0] < 100:
                # Skip views with too few points
                continue

            # Normalize point cloud
            points = normalize_point_cloud(points)

            # Sample fixed number of points
            points = sample_point_cloud(points, config.POINTNET_NUM_POINTS)

            # Extract vector (camera pose)
            position = cameras[view_id]['position']
            rotation = cameras[view_id]['rotation']
            vector = position + rotation  # [x, y, z, qx, qy, qz, qw]

            # Pad vector to 15D (match MultiModalNet)
            vector = vector + [0.0] * (15 - len(vector))

            point_clouds.append(points)
            vectors.append(vector)

        if len(point_clouds) == 0:
            skipped_count += 1
            continue

        # Save processed sample
        output_sample_dir = output_dir / sample_name
        output_sample_dir.mkdir(exist_ok=True)

        # Save point clouds as numpy array [num_views, num_points, 3]
        point_clouds = np.stack(point_clouds, axis=0).astype(np.float32)
        np.save(output_sample_dir / "point_clouds.npy", point_clouds)

        # Save vectors as numpy array [num_views, 15]
        vectors = np.array(vectors, dtype=np.float32)
        np.save(output_sample_dir / "vectors.npy", vectors)

        # Save metadata
        metadata = {
            "class_id": target_class,
            "num_views": len(point_clouds),
            "num_points": config.POINTNET_NUM_POINTS
        }
        with open(output_sample_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        processed_count += 1

    print(f"\n{'='*60}")
    print(f"Dataset generation complete!")
    print(f"Processed: {processed_count} samples")
    print(f"Skipped: {skipped_count} samples")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    generate_pointnet_dataset()
