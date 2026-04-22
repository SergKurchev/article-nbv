"""Test script for scene generation stages.

This script tests all three scene generation stages:
- Stage 1: Single object, no obstacles
- Stage 2: Multiple objects (2-10), no obstacles
- Stage 3: Multiple objects (2-10) + obstacles

Usage:
    uv run python scripts/test_scene_stages.py --stage 1
    uv run python scripts/test_scene_stages.py --stage 2
    uv run python scripts/test_scene_stages.py --stage 3
    uv run python scripts/test_scene_stages.py --all  # Test all stages
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pybullet as p
import pybullet_data
import time
import config
from src.simulation.asset_loader import AssetLoader


def test_stage(stage_num, duration=10):
    """Test a specific scene stage.

    Args:
        stage_num: Stage number (1, 2, or 3)
        duration: How long to display the scene (seconds)
    """
    print(f"\n{'='*60}")
    print(f"Testing Stage {stage_num}")
    print(f"{'='*60}")

    # Temporarily override config
    original_stage = config.SCENE_STAGE
    config.SCENE_STAGE = stage_num

    # Connect to PyBullet with GUI
    client_id = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81, physicsClientId=client_id)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)

    # Load plane
    p.loadURDF("plane.urdf", physicsClientId=client_id)

    # Configure camera view
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0.5, 0.0, 0.2],
        physicsClientId=client_id
    )

    # Generate scene
    asset_loader = AssetLoader(client_id)
    target_objects = asset_loader.generate_scene()

    # Print scene info
    print(f"\nScene Configuration:")
    print(f"  Stage: {stage_num}")
    print(f"  Number of objects: {len(target_objects)}")
    print(f"  Number of obstacles: {len(asset_loader.obstacles)}")

    if stage_num == 1:
        print(f"\n  Expected: 1 object at [0.5, 0.0, 0.2], 0 obstacles")
    elif stage_num == 2:
        print(f"\n  Expected: {config.MIN_OBJECTS}-{config.MAX_OBJECTS} objects, 0 obstacles")
        print(f"  Spatial bounds:")
        print(f"    X: [{config.SCENE_BOUNDS_X_MIN}, {config.SCENE_BOUNDS_X_MAX}]")
        print(f"    Y: [{config.SCENE_BOUNDS_Y_MIN}, {config.SCENE_BOUNDS_Y_MAX}]")
        print(f"    Z: [{config.SCENE_BOUNDS_Z_MIN}, {config.SCENE_BOUNDS_Z_MAX}]")
    elif stage_num == 3:
        print(f"\n  Expected: {config.MIN_OBJECTS}-{config.MAX_OBJECTS} objects, {config.MIN_OBSTACLES}-{config.MAX_OBSTACLES} obstacles")
        print(f"  Spatial bounds:")
        print(f"    X: [{config.SCENE_BOUNDS_X_MIN}, {config.SCENE_BOUNDS_X_MAX}]")
        print(f"    Y: [{config.SCENE_BOUNDS_Y_MIN}, {config.SCENE_BOUNDS_Y_MAX}]")
        print(f"    Z: [{config.SCENE_BOUNDS_Z_MIN}, {config.SCENE_BOUNDS_Z_MAX}]")

    # Get object positions
    print(f"\n  Object positions:")
    for i, obj_id in enumerate(target_objects):
        pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=client_id)
        print(f"    Object {i+1}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # Get obstacle positions
    if asset_loader.obstacles:
        print(f"\n  Obstacle positions:")
        for i, obs_id in enumerate(asset_loader.obstacles):
            pos, _ = p.getBasePositionAndOrientation(obs_id, physicsClientId=client_id)
            print(f"    Obstacle {i+1}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # Verify collision-free placement
    print(f"\n  Verifying collision-free placement...")
    collision_found = False

    # Check object-object distances
    for i, obj1_id in enumerate(target_objects):
        pos1, _ = p.getBasePositionAndOrientation(obj1_id, physicsClientId=client_id)
        for j, obj2_id in enumerate(target_objects[i+1:], start=i+1):
            pos2, _ = p.getBasePositionAndOrientation(obj2_id, physicsClientId=client_id)
            distance = ((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2 + (pos1[2]-pos2[2])**2)**0.5
            if distance < config.SCENE_MIN_OBJECT_DISTANCE:
                print(f"    WARNING: Objects {i+1} and {j+1} too close: {distance:.3f}m")
                collision_found = True

    if not collision_found:
        print(f"    ✓ All objects are collision-free (min distance: {config.SCENE_MIN_OBJECT_DISTANCE}m)")

    print(f"\n  Displaying scene for {duration} seconds...")
    print(f"  (Close the window or press Ctrl+C to exit early)")

    # Display scene
    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            p.stepSimulation(physicsClientId=client_id)
            time.sleep(1/240)
    except KeyboardInterrupt:
        print("\n  Interrupted by user")

    # Cleanup
    p.disconnect(physicsClientId=client_id)
    config.SCENE_STAGE = original_stage

    print(f"\n{'='*60}")
    print(f"Stage {stage_num} test complete")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Test scene generation stages")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], help="Stage to test (1, 2, or 3)")
    parser.add_argument("--all", action="store_true", help="Test all stages sequentially")
    parser.add_argument("--duration", type=int, default=10, help="Display duration per stage (seconds)")

    args = parser.parse_args()

    if args.all:
        for stage in [1, 2, 3]:
            test_stage(stage, args.duration)
            time.sleep(1)
    elif args.stage:
        test_stage(args.stage, args.duration)
    else:
        print("Error: Must specify --stage or --all")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
