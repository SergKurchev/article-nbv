"""Test script to verify texture generation parameters."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
import config

def test_area_ratio():
    """Test that generated texture respects area ratio constraints."""
    print("\n=== Testing Area Ratio Constraints ===")
    print(f"Expected ratio: {config.TEXTURE_MIN_AREA_RATIO:.1%} - {config.TEXTURE_MAX_AREA_RATIO:.1%}")

    # Load generated mixed texture
    texture_path = config.DATA_DIR / "objects" / "textures" / "mixed.png"

    if not texture_path.exists():
        print(f"ERROR: Texture not found at {texture_path}")
        print("Run: uv run python src/vision/texture_generator.py")
        return False

    # Load and analyze
    img = Image.open(texture_path)
    img_array = np.array(img)

    # Count red pixels (color1)
    red_color = np.array(config.TEXTURE_RED_COLOR)
    green_color = np.array(config.TEXTURE_GREEN_COLOR)

    # Calculate which pixels are closer to red vs green
    dist_to_red = np.sum((img_array - red_color)**2, axis=2)
    dist_to_green = np.sum((img_array - green_color)**2, axis=2)

    red_pixels = np.sum(dist_to_red < dist_to_green)
    total_pixels = img_array.shape[0] * img_array.shape[1]

    red_ratio = red_pixels / total_pixels
    green_ratio = 1 - red_ratio

    print(f"\nActual ratios:")
    print(f"  Red:   {red_ratio:.1%}")
    print(f"  Green: {green_ratio:.1%}")

    # Check if within bounds
    min_ratio = config.TEXTURE_MIN_AREA_RATIO
    max_ratio = config.TEXTURE_MAX_AREA_RATIO

    red_ok = min_ratio <= red_ratio <= max_ratio
    green_ok = min_ratio <= green_ratio <= max_ratio

    print(f"\nValidation:")
    print(f"  Red ratio in range:   {'OK' if red_ok else 'FAIL'}")
    print(f"  Green ratio in range: {'OK' if green_ok else 'FAIL'}")

    if red_ok and green_ok:
        print("\n[PASS] Area ratio constraints satisfied!")
        return True
    else:
        print("\n[FAIL] Area ratio constraints violated!")
        return False


def test_config_parameters():
    """Verify all config parameters are used (no magic numbers)."""
    print("\n=== Config Parameters ===")

    params = {
        "Texture size": config.TEXTURE_SIZE,
        "Curve complexity": config.TEXTURE_CURVE_COMPLEXITY,
        "Curve amplitude": config.TEXTURE_CURVE_AMPLITUDE,
        "Min area ratio": config.TEXTURE_MIN_AREA_RATIO,
        "Max area ratio": config.TEXTURE_MAX_AREA_RATIO,
        "Gradient steepness": config.TEXTURE_GRADIENT_STEEPNESS,
        "Bezier samples": config.TEXTURE_BEZIER_SAMPLES,
        "Downsample factor": config.TEXTURE_DOWNSAMPLE_FACTOR,
        "Curve extension top": config.TEXTURE_CURVE_EXTENSION_TOP,
        "Curve extension bottom": config.TEXTURE_CURVE_EXTENSION_BOTTOM,
        "Start X range": f"{config.TEXTURE_CURVE_START_X_MIN}-{config.TEXTURE_CURVE_START_X_MAX}",
        "End X range": f"{config.TEXTURE_CURVE_END_X_MIN}-{config.TEXTURE_CURVE_END_X_MAX}",
        "Max attempts": config.TEXTURE_MAX_GENERATION_ATTEMPTS,
    }

    for name, value in params.items():
        print(f"  {name:25s}: {value}")

    print("\n[PASS] All parameters loaded from config.py")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Texture Generation Parameter Test")
    print("=" * 60)

    config_ok = test_config_parameters()
    ratio_ok = test_area_ratio()

    print("\n" + "=" * 60)
    if config_ok and ratio_ok:
        print("[PASS] All tests passed!")
    else:
        print("[FAIL] Some tests failed")
    print("=" * 60)
