"""Analyze PyBullet simulation screenshots for ground clearance validation."""
import sys
from pathlib import Path
from PIL import Image
import numpy as np

def analyze_screenshot(image_path, stage_name):
    """Analyze a single screenshot for object placement."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {stage_name}")
    print(f"{'='*60}")

    img = Image.open(image_path)
    img_array = np.array(img)

    print(f"Image size: {img.size}")
    print(f"Image mode: {img.mode}")

    # Basic color analysis
    height, width = img_array.shape[:2]

    # Check bottom 20% of image for ground plane
    ground_region = img_array[int(height*0.8):, :]

    # Check middle region for objects
    object_region = img_array[int(height*0.3):int(height*0.8), :]

    print(f"\nVisual inspection required:")
    print(f"- Check if objects have visible shadows/gaps below them")
    print(f"- Verify no objects are touching the ground plane")
    print(f"- Look for any object-object collisions")

    return img

def main():
    base_dir = Path(__file__).parent
    screenshots_dir = base_dir / "screenshots"

    stages = [
        ("stage_1_validation.png", "Stage 1: Single Object"),
        ("stage_2_validation.png", "Stage 2: Multiple Objects"),
        ("stage_3_validation.png", "Stage 3: Multiple Objects + Obstacles"),
    ]

    print("PYBULLET SIMULATION VALIDATION REPORT")
    print("="*60)

    for filename, stage_name in stages:
        image_path = screenshots_dir / filename
        if image_path.exists():
            analyze_screenshot(image_path, stage_name)
        else:
            print(f"\n[ERROR] {stage_name}: File not found - {image_path}")

    print("\n" + "="*60)
    print("MANUAL VALIDATION REQUIRED")
    print("="*60)
    print("\nPlease visually inspect the screenshots and verify:")
    print("1. ALL objects are floating with visible clearance")
    print("2. ALL obstacles (if any) are floating with visible clearance")
    print("3. No object-object or object-obstacle collisions")
    print("\nScreenshots location:")
    print(f"  {screenshots_dir.absolute()}")

if __name__ == "__main__":
    main()
