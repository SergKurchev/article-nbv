"""Automated screenshot capture and validation for scene stages.

This script captures screenshots of each scene stage and validates them
using a vision model to ensure correct implementation.

Usage:
    uv run python scripts/validate_stages.py
    uv run python scripts/validate_stages.py --stage 2
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pybullet as p
import pybullet_data
import numpy as np
import config
from src.simulation.asset_loader import AssetLoader
import argparse
from PIL import Image
import time


def capture_screenshot(client_id, width=800, height=600, filename="screenshot.png"):
    """Capture screenshot from PyBullet simulation.

    Args:
        client_id: PyBullet client ID
        width: Image width
        height: Image height
        filename: Output filename

    Returns:
        PIL.Image: Captured image
    """
    # Get camera view matrix
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[1.5, 0.0, 0.8],
        cameraTargetPosition=[0.5, 0.0, 0.2],
        cameraUpVector=[0, 0, 1],
        physicsClientId=client_id
    )

    # Get projection matrix
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=width / height,
        nearVal=0.1,
        farVal=10.0
    )

    # Capture image
    width, height, rgb, depth, seg = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        physicsClientId=client_id
    )

    # Convert to PIL Image
    rgb_array = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)
    rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
    image = Image.fromarray(rgb_array)

    # Save image
    output_path = Path(__file__).parent.parent / "screenshots" / filename
    output_path.parent.mkdir(exist_ok=True)
    image.save(output_path)

    print(f"Screenshot saved: {output_path}")

    return image, str(output_path)


def setup_scene(stage_num):
    """Setup PyBullet scene for given stage.

    Args:
        stage_num: Stage number (1, 2, or 3)

    Returns:
        tuple: (client_id, asset_loader, target_objects)
    """
    # Temporarily override config
    original_stage = config.SCENE_STAGE
    config.SCENE_STAGE = stage_num

    # Connect to PyBullet (headless)
    client_id = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=client_id)

    # Add pybullet_data to search path
    p.setAdditionalSearchPath(config.get_short_path(pybullet_data.getDataPath()), physicsClientId=client_id)

    # Load plane
    p.loadURDF("plane.urdf", physicsClientId=client_id)

    # Generate scene
    asset_loader = AssetLoader(client_id)
    target_objects = asset_loader.generate_scene()

    # Wait for settling
    for _ in range(10):
        p.stepSimulation(physicsClientId=client_id)

    return client_id, asset_loader, target_objects


def validate_stage_with_agent(stage_num, image_path):
    """Validate stage screenshot using AI agent.

    Args:
        stage_num: Stage number (1, 2, or 3)
        image_path: Path to screenshot

    Returns:
        dict: Validation results
    """
    # Expected characteristics for each stage
    expected = {
        1: {
            "num_objects": 1,
            "num_obstacles": 0,
            "description": "Single object at fixed position [0.5, 0.0, 0.2], no obstacles"
        },
        2: {
            "num_objects": (2, 10),
            "num_obstacles": 0,
            "description": "Multiple objects (2-10) with uniform spatial distribution, no obstacles"
        },
        3: {
            "num_objects": (2, 10),
            "num_obstacles": (1, 5),
            "description": "Multiple objects (2-10) + obstacles (1-5) with uniform spatial distribution"
        }
    }

    exp = expected[stage_num]

    # Validation prompt for agent
    validation_prompt = f"""Analyze this PyBullet simulation screenshot for Stage {stage_num} validation.

Expected configuration:
- Stage {stage_num}: {exp['description']}
- Number of objects: {exp['num_objects']}
- Number of obstacles: {exp['num_obstacles']}

Please analyze the image and answer:
1. How many colored objects (primitives) do you see? (Count distinct 3D shapes with textures)
2. How many gray obstacles (thin panels) do you see?
3. Are objects spatially distributed (not overlapping)?
4. Does the scene match the expected Stage {stage_num} configuration?
5. Any issues or anomalies?

Provide a structured response with:
- Object count: [number]
- Obstacle count: [number]
- Spatial distribution: [good/poor/overlapping]
- Stage match: [yes/no]
- Issues: [list any problems]
- Overall validation: [PASS/FAIL]
"""

    return {
        "stage": stage_num,
        "image_path": image_path,
        "expected": exp,
        "validation_prompt": validation_prompt,
        "status": "pending"  # Will be filled by agent
    }


def capture_and_validate_stage(stage_num):
    """Capture screenshot and prepare validation for a stage.

    Args:
        stage_num: Stage number (1, 2, or 3)

    Returns:
        dict: Validation results
    """
    print(f"\n{'='*60}")
    print(f"Stage {stage_num}: Capture and Validation")
    print(f"{'='*60}")

    # Setup scene
    print(f"Setting up Stage {stage_num} scene...")
    client_id, asset_loader, target_objects = setup_scene(stage_num)

    # Get scene info
    num_objects = len(target_objects)
    num_obstacles = len(asset_loader.obstacles)

    print(f"Scene generated:")
    print(f"  Objects: {num_objects}")
    print(f"  Obstacles: {num_obstacles}")

    # Capture screenshot
    print(f"Capturing screenshot...")
    image, image_path = capture_screenshot(
        client_id,
        filename=f"stage_{stage_num}_validation.png"
    )

    # Cleanup
    p.disconnect(physicsClientId=client_id)

    # Prepare validation
    validation = validate_stage_with_agent(stage_num, image_path)
    validation["actual_objects"] = num_objects
    validation["actual_obstacles"] = num_obstacles

    print(f"\nScreenshot captured: {image_path}")
    print(f"Validation prompt prepared")

    return validation


def main():
    parser = argparse.ArgumentParser(description="Validate scene stages with screenshots")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], help="Stage to validate")
    parser.add_argument("--all", action="store_true", help="Validate all stages")

    args = parser.parse_args()

    print("="*60)
    print("Scene Stage Validation with Screenshots")
    print("="*60)

    validations = []

    if args.all:
        stages = [1, 2, 3]
    elif args.stage:
        stages = [args.stage]
    else:
        print("Error: Must specify --stage or --all")
        parser.print_help()
        return

    # Capture screenshots for all stages
    for stage in stages:
        validation = capture_and_validate_stage(stage)
        validations.append(validation)
        time.sleep(1)

    # Print summary
    print(f"\n{'='*60}")
    print("Validation Summary")
    print(f"{'='*60}")

    for val in validations:
        print(f"\nStage {val['stage']}:")
        print(f"  Screenshot: {val['image_path']}")
        print(f"  Objects: {val['actual_objects']} (expected: {val['expected']['num_objects']})")
        print(f"  Obstacles: {val['actual_obstacles']} (expected: {val['expected']['num_obstacles']})")

    print(f"\n{'='*60}")
    print("Next Steps:")
    print(f"{'='*60}")
    print("\nScreenshots have been captured. To validate with AI agent:")
    print("\n1. Review screenshots in: screenshots/")
    print("2. Use the validation prompts below with an AI vision model")
    print("3. Or manually inspect the screenshots")

    print("\n" + "="*60)
    print("Validation Prompts for AI Agent:")
    print("="*60)

    for val in validations:
        print(f"\n--- Stage {val['stage']} ---")
        print(f"Image: {val['image_path']}")
        print(f"\nPrompt:")
        print(val['validation_prompt'])

    return validations


if __name__ == "__main__":
    main()
