"""Procedural texture generation for object classification.

Generates three texture types:
- red: Fully red texture (ripe objects)
- mixed: Red-green curved gradient (partially ripe)
- green: Fully green texture (unripe)
"""

import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from PIL import Image
import config


def generate_solid_texture(color, size):
    """Generate solid color texture.

    Args:
        color: RGB tuple (e.g., [255, 0, 0])
        size: Texture resolution (e.g., 512)

    Returns:
        PIL.Image: Generated texture
    """
    texture = np.zeros((size, size, 3), dtype=np.uint8)
    texture[:, :] = color
    return Image.fromarray(texture)


def generate_bezier_control_points(complexity, amplitude, size):
    """Generate random Bezier curve control points.

    Args:
        complexity: Number of control points (2-5)
        amplitude: Maximum curve deviation (0.0-0.5)
        size: Texture resolution

    Returns:
        np.ndarray: Control points array of shape (complexity, 2)
    """
    complexity = max(2, min(5, complexity))

    # Start and end points on opposite edges
    if np.random.rand() < 0.5:
        # Vertical orientation
        start = [0, np.random.uniform(0.2, 0.8) * size]
        end = [size, np.random.uniform(0.2, 0.8) * size]
    else:
        # Horizontal orientation
        start = [np.random.uniform(0.2, 0.8) * size, 0]
        end = [np.random.uniform(0.2, 0.8) * size, size]

    # Generate intermediate control points
    control_points = [start]

    for i in range(1, complexity - 1):
        t = i / (complexity - 1)
        base_x = start[0] + t * (end[0] - start[0])
        base_y = start[1] + t * (end[1] - start[1])

        # Add random deviation perpendicular to line
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)

        if length > 0:
            perp_x = -dy / length
            perp_y = dx / length
        else:
            perp_x, perp_y = 0, 0

        deviation = np.random.uniform(-amplitude, amplitude) * size

        control_points.append([
            base_x + perp_x * deviation,
            base_y + perp_y * deviation
        ])

    control_points.append(end)

    return np.array(control_points)


def evaluate_bezier_curve(control_points, num_samples=1000):
    """Evaluate Bezier curve at multiple points.

    Args:
        control_points: Array of control points (N, 2)
        num_samples: Number of points to sample along curve

    Returns:
        np.ndarray: Curve points array of shape (num_samples, 2)
    """
    n = len(control_points) - 1
    t_values = np.linspace(0, 1, num_samples)
    curve_points = []

    for t in t_values:
        # De Casteljau's algorithm for Bezier curve evaluation
        points = control_points.copy()

        for r in range(1, n + 1):
            for i in range(n - r + 1):
                points[i] = (1 - t) * points[i] + t * points[i + 1]

        curve_points.append(points[0])

    return np.array(curve_points)


def compute_distance_field(control_points, size):
    """Compute signed distance field from Bezier curve.

    Args:
        control_points: Array of control points (N, 2)
        size: Texture resolution

    Returns:
        np.ndarray: Distance field of shape (size, size)
    """
    # Evaluate curve at many points
    curve_points = evaluate_bezier_curve(control_points, num_samples=2000)

    # Create coordinate grid
    y, x = np.mgrid[0:size, 0:size]
    coords = np.stack([x, y], axis=-1)

    # Compute minimum distance to curve for each pixel
    distances = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            point = coords[i, j]
            dists = np.sqrt(np.sum((curve_points - point)**2, axis=1))
            distances[i, j] = np.min(dists)

    # Determine sign (which side of curve)
    # Use cross product to determine side
    curve_start = curve_points[0]
    curve_end = curve_points[-1]
    curve_vec = curve_end - curve_start

    signs = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            point = coords[i, j]
            point_vec = point - curve_start
            cross = curve_vec[0] * point_vec[1] - curve_vec[1] * point_vec[0]
            signs[i, j] = 1 if cross > 0 else -1

    return distances * signs


def validate_area_ratio(distance_field, min_ratio, max_ratio):
    """Check if area ratio constraints are satisfied.

    Args:
        distance_field: Signed distance field
        min_ratio: Minimum area ratio (e.g., 0.3)
        max_ratio: Maximum area ratio (e.g., 0.7)

    Returns:
        bool: True if constraints satisfied
    """
    total_pixels = distance_field.size
    positive_pixels = np.sum(distance_field > 0)
    ratio = positive_pixels / total_pixels

    return min_ratio <= ratio <= max_ratio


def apply_gradient(distance_field, color1, color2, sharpness):
    """Apply gradient based on distance field.

    Args:
        distance_field: Signed distance field
        color1: RGB tuple for negative side
        color2: RGB tuple for positive side
        sharpness: Gradient transition sharpness (0.0-1.0)

    Returns:
        np.ndarray: RGB texture array
    """
    size = distance_field.shape[0]

    # Normalize distances for gradient
    max_dist = np.max(np.abs(distance_field))
    if max_dist > 0:
        normalized = distance_field / max_dist
    else:
        normalized = distance_field

    # Use moderate steepness for visible but clear gradient
    # Too steep (20) = almost no gradient visible
    # Too soft (5) = muddy colors
    # Sweet spot: 8-10
    steepness = 8
    blend = 1 / (1 + np.exp(-steepness * normalized))

    # Create RGB texture
    texture = np.zeros((size, size, 3), dtype=np.uint8)

    # Apply gradient per channel
    for c in range(3):
        texture[:, :, c] = (
            color1[c] * (1 - blend) + color2[c] * blend
        ).astype(np.uint8)

    return texture


def add_noise(texture, noise_level):
    """Add realistic noise to texture.

    Args:
        texture: PIL.Image or numpy array
        noise_level: Noise strength (0.0-0.2)

    Returns:
        PIL.Image: Noisy texture
    """
    if isinstance(texture, Image.Image):
        texture_array = np.array(texture)
    else:
        texture_array = texture

    if noise_level <= 0:
        return Image.fromarray(texture_array) if isinstance(texture, np.ndarray) else texture

    noise = np.random.normal(0, noise_level * 255, texture_array.shape)
    noisy = np.clip(texture_array.astype(float) + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy)


def generate_curved_gradient_texture(color1, color2, complexity, amplitude, sharpness, size, min_ratio, max_ratio, max_attempts=10):
    """Generate texture with curved gradient boundary.

    Args:
        color1: RGB tuple for first color
        color2: RGB tuple for second color
        complexity: Number of Bezier control points (2-5)
        amplitude: Maximum curve deviation (0.0-0.5)
        sharpness: Gradient transition sharpness (0.0-1.0)
        size: Texture resolution
        min_ratio: Minimum area for each color (e.g., 0.3)
        max_ratio: Maximum area for each color (e.g., 0.7)
        max_attempts: Maximum generation attempts

    Returns:
        PIL.Image: Generated texture

    Raises:
        RuntimeError: If unable to generate valid texture after max_attempts
    """
    for attempt in range(max_attempts):
        # Generate random Bezier curve
        control_points = generate_bezier_control_points(complexity, amplitude, size)

        # Create distance field from curve
        distance_field = compute_distance_field(control_points, size)

        # Validate area ratio constraints
        if validate_area_ratio(distance_field, min_ratio, max_ratio):
            # Apply gradient based on distance
            texture_array = apply_gradient(distance_field, color1, color2, sharpness)
            return Image.fromarray(texture_array)

    raise RuntimeError(f"Failed to generate valid curved gradient after {max_attempts} attempts")


def generate_all_textures(output_dir=None, num_variants=3):
    """Generate all texture variants with multiple random mixed textures.

    Args:
        output_dir: Output directory path (default: src/data/objects/textures/)
        num_variants: Number of random mixed texture variants to generate

    Returns:
        dict: Mapping of texture_type to file path
    """
    if output_dir is None:
        output_dir = config.DATA_DIR / "objects" / "textures"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texture_paths = {}

    # Generate red texture
    print("Generating red texture...")
    red_texture = generate_solid_texture(
        config.TEXTURE_RED_COLOR,
        config.TEXTURE_SIZE
    )
    red_texture = add_noise(red_texture, config.TEXTURE_NOISE_LEVEL)
    red_path = output_dir / "red.png"
    red_texture.save(red_path)
    texture_paths["red"] = red_path
    print(f"  Saved to {red_path}")

    # Generate multiple mixed textures with different random curves
    print(f"Generating {num_variants} mixed texture variants...")
    for variant_idx in range(num_variants):
        mixed_texture = generate_curved_gradient_texture(
            config.TEXTURE_RED_COLOR,
            config.TEXTURE_GREEN_COLOR,
            config.TEXTURE_CURVE_COMPLEXITY,
            config.TEXTURE_CURVE_AMPLITUDE,
            config.TEXTURE_GRADIENT_SHARPNESS,
            config.TEXTURE_SIZE,
            config.TEXTURE_MIN_AREA_RATIO,
            config.TEXTURE_MAX_AREA_RATIO
        )
        mixed_texture = add_noise(mixed_texture, config.TEXTURE_NOISE_LEVEL)

        if variant_idx == 0:
            mixed_path = output_dir / "mixed.png"
        else:
            mixed_path = output_dir / f"mixed_{variant_idx}.png"

        mixed_texture.save(mixed_path)
        texture_paths[f"mixed_{variant_idx}"] = mixed_path
        print(f"  Saved variant {variant_idx} to {mixed_path}")

    # Generate green texture
    print("Generating green texture...")
    green_texture = generate_solid_texture(
        config.TEXTURE_GREEN_COLOR,
        config.TEXTURE_SIZE
    )
    green_texture = add_noise(green_texture, config.TEXTURE_NOISE_LEVEL)
    green_path = output_dir / "green.png"
    green_texture.save(green_path)
    texture_paths["green"] = green_path
    print(f"  Saved to {green_path}")

    return texture_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate procedural textures for NBV project")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Show generated textures")
    args = parser.parse_args()

    # Generate textures
    texture_paths = generate_all_textures(args.output)

    print("\nTexture generation complete!")
    print(f"Generated {len(texture_paths)} textures:")
    for texture_type, path in texture_paths.items():
        print(f"  {texture_type}: {path}")

    # Visualize if requested
    if args.visualize:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for ax, (texture_type, path) in zip(axes, texture_paths.items()):
            img = Image.open(path)
            ax.imshow(img)
            ax.set_title(f"{texture_type.capitalize()} Texture")
            ax.axis("off")

        plt.tight_layout()
        plt.show()
