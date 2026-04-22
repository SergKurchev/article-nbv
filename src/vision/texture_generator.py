"""Ultra-fast texture generator using optimized distance field with Numba JIT."""

import numpy as np
from PIL import Image, ImageDraw
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback: no-op decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


def generate_bezier_control_points(complexity, amplitude, size, start_x_min, start_x_max,
                                   end_x_min, end_x_max, extension_top, extension_bottom):
    """Generate random Bezier curve control points with random orientation.

    Args:
        complexity: Number of control points (2-10)
        amplitude: Maximum curve deviation (0.0-0.5)
        size: Texture size in pixels
        start_x_min: Minimum horizontal position for curve start (0.0-1.0)
        start_x_max: Maximum horizontal position for curve start (0.0-1.0)
        end_x_min: Minimum horizontal position for curve end (0.0-1.0)
        end_x_max: Maximum horizontal position for curve end (0.0-1.0)
        extension_top: Extend curve beyond top edge (fraction of size)
        extension_bottom: Extend curve beyond bottom edge (fraction of size)

    Returns:
        np.ndarray: Control points array of shape (complexity, 2)
    """
    complexity = max(2, min(10, complexity))

    # Random angle for curve orientation (0 to 360 degrees)
    angle = np.random.uniform(0, 2 * np.pi)

    # Generate curve along vertical direction first (from edge to edge)
    start = [np.random.uniform(start_x_min, start_x_max) * size, -size * extension_top]
    end = [np.random.uniform(end_x_min, end_x_max) * size, size * (1 + extension_bottom)]

    control_points = [start]

    for i in range(1, complexity - 1):
        t = i / (complexity - 1)
        base_x = start[0] + t * (end[0] - start[0])
        base_y = start[1] + t * (end[1] - start[1])

        # Add deviation perpendicular to the line
        deviation = np.random.uniform(-amplitude, amplitude) * size

        control_points.append([base_x + deviation, base_y])

    control_points.append(end)

    control_points = np.array(control_points)

    # Rotate around center
    center = np.array([size / 2, size / 2])
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    # Translate to origin, rotate, translate back
    control_points = (control_points - center) @ rotation_matrix.T + center

    return control_points


def evaluate_bezier_curve_fast(control_points, num_samples=500):
    """Fast Bezier curve evaluation using matrix operations."""
    n = len(control_points) - 1
    t = np.linspace(0, 1, num_samples)

    # Compute binomial coefficients
    from math import comb
    coeffs = np.array([comb(n, i) for i in range(n + 1)])

    # Compute Bernstein polynomials
    bernstein = np.zeros((num_samples, n + 1))
    for i in range(n + 1):
        bernstein[:, i] = coeffs[i] * (t ** i) * ((1 - t) ** (n - i))

    # Compute curve points
    curve_points = bernstein @ control_points

    return curve_points


@jit(nopython=True, cache=True)
def compute_distances_numba(pixels, curve_points):
    """Numba-optimized distance computation.

    Args:
        pixels: Array of pixel coordinates (N, 2)
        curve_points: Array of curve points (M, 2)

    Returns:
        closest_indices: Index of closest curve point for each pixel (N,)
        min_distances: Distance to closest curve point for each pixel (N,)
    """
    n_pixels = pixels.shape[0]
    n_curve = curve_points.shape[0]

    closest_indices = np.zeros(n_pixels, dtype=np.int32)
    min_distances = np.zeros(n_pixels, dtype=np.float32)

    for i in range(n_pixels):
        min_dist = 1e10
        closest_idx = 0

        for j in range(n_curve):
            dx = pixels[i, 0] - curve_points[j, 0]
            dy = pixels[i, 1] - curve_points[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)

            if dist < min_dist:
                min_dist = dist
                closest_idx = j

        closest_indices[i] = closest_idx
        min_distances[i] = min_dist

    return closest_indices, min_distances


@jit(nopython=True, cache=True)
def compute_signed_distances_numba(pixels, curve_points, normals, closest_indices, min_distances):
    """Numba-optimized signed distance computation.

    Args:
        pixels: Pixel coordinates (N, 2)
        curve_points: Curve points (M, 2)
        normals: Normal vectors at curve points (M, 2)
        closest_indices: Closest curve point indices (N,)
        min_distances: Distances to closest points (N,)

    Returns:
        signed_distances: Signed distances (N,)
    """
    n_pixels = pixels.shape[0]
    signed_distances = np.zeros(n_pixels, dtype=np.float32)

    for i in range(n_pixels):
        idx = closest_indices[i]

        # Vector from curve point to pixel
        to_pixel_x = pixels[i, 0] - curve_points[idx, 0]
        to_pixel_y = pixels[i, 1] - curve_points[idx, 1]

        # Dot product with normal
        dot = to_pixel_x * normals[idx, 0] + to_pixel_y * normals[idx, 1]

        # Sign
        sign = 1.0 if dot >= 0 else -1.0

        signed_distances[i] = sign * min_distances[i]

    return signed_distances


def create_gradient_fast(curve_points, size, color1, color2, steepness, downsample_factor):
    """Fast gradient using optimized distance field with downsampling and Numba JIT.

    Args:
        curve_points: Evaluated Bezier curve points
        size: Texture size in pixels
        color1: RGB color for one side of gradient
        color2: RGB color for other side of gradient
        steepness: Sigmoid steepness for transition sharpness
        downsample_factor: Downsample factor for speed optimization

    Returns:
        np.ndarray: RGB texture array of shape (size, size, 3)
    """
    small_size = size // downsample_factor

    # Create downsampled grid
    y, x = np.mgrid[0:small_size, 0:small_size]
    pixels = np.stack([x.ravel() * downsample_factor, y.ravel() * downsample_factor], axis=1).astype(np.float32)

    # Convert curve points to float32 for numba
    curve_points_f32 = curve_points.astype(np.float32)

    # Compute distances to curve (Numba-optimized if available)
    if NUMBA_AVAILABLE:
        closest_indices, min_distances = compute_distances_numba(pixels, curve_points_f32)
    else:
        # Fallback: vectorized numpy
        diff = pixels[:, np.newaxis, :] - curve_points_f32[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        closest_indices = np.argmin(distances, axis=1)
        min_distances = distances[np.arange(len(pixels)), closest_indices]

    # Compute tangents at curve points
    tangents = np.zeros_like(curve_points_f32)
    tangents[1:-1] = curve_points_f32[2:] - curve_points_f32[:-2]
    tangents[0] = curve_points_f32[1] - curve_points_f32[0]
    tangents[-1] = curve_points_f32[-1] - curve_points_f32[-2]

    # Normalize tangents
    tangent_lengths = np.sqrt(np.sum(tangents**2, axis=1, keepdims=True))
    tangent_lengths = np.maximum(tangent_lengths, 1e-8)
    tangents = tangents / tangent_lengths

    # Compute normals (perpendicular to tangents)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1).astype(np.float32)

    # Compute signed distances (Numba-optimized if available)
    if NUMBA_AVAILABLE:
        signed_distances = compute_signed_distances_numba(
            pixels, curve_points_f32, normals, closest_indices, min_distances
        )
    else:
        # Fallback: vectorized numpy
        closest_normals = normals[closest_indices]
        closest_curve_points = curve_points_f32[closest_indices]
        to_pixel = pixels - closest_curve_points
        signs = np.sign(np.sum(to_pixel * closest_normals, axis=1))
        signs[signs == 0] = 1
        signed_distances = signs * min_distances

    signed_distances = signed_distances.reshape(small_size, small_size)

    # Normalize distances
    max_dist = np.max(np.abs(signed_distances))
    if max_dist > 0:
        normalized = signed_distances / max_dist
    else:
        normalized = signed_distances

    # Apply sigmoid for smooth transition
    blend = 1 / (1 + np.exp(-steepness * normalized))

    # Create RGB texture (downsampled)
    texture_small = np.zeros((small_size, small_size, 3), dtype=np.uint8)
    color1 = np.array(color1)
    color2 = np.array(color2)
    for c in range(3):
        texture_small[:, :, c] = (
            color1[c] * (1 - blend) + color2[c] * blend
        ).astype(np.uint8)

    # Upsample to full size using PIL (fast and smooth)
    texture_img = Image.fromarray(texture_small)
    texture_img = texture_img.resize((size, size), Image.BILINEAR)
    texture_array = np.array(texture_img)

    return texture_array


def generate_curved_gradient_texture(
    color1, color2, complexity, amplitude, size,
    min_ratio, max_ratio, steepness,
    start_x_min, start_x_max, end_x_min, end_x_max,
    extension_top, extension_bottom, bezier_samples, downsample_factor,
    max_attempts=10, visualize=False
):
    """Generate texture with curved gradient boundary.

    Args:
        color1: RGB color for one side
        color2: RGB color for other side
        complexity: Number of Bezier control points
        amplitude: Maximum curve deviation
        size: Texture size in pixels
        min_ratio: Minimum area ratio for color balance
        max_ratio: Maximum area ratio for color balance
        steepness: Sigmoid steepness for gradient transition
        start_x_min: Min horizontal position for curve start
        start_x_max: Max horizontal position for curve start
        end_x_min: Min horizontal position for curve end
        end_x_max: Max horizontal position for curve end
        extension_top: Extend curve beyond top edge
        extension_bottom: Extend curve beyond bottom edge
        bezier_samples: Number of points to sample along curve
        downsample_factor: Downsample factor for optimization
        max_attempts: Maximum generation attempts
        visualize: Whether to generate debug visualization

    Returns:
        PIL.Image or tuple: Generated texture, optionally with debug image
    """
    for attempt in range(max_attempts):
        # Generate random Bezier curve
        control_points = generate_bezier_control_points(
            complexity, amplitude, size,
            start_x_min, start_x_max, end_x_min, end_x_max,
            extension_top, extension_bottom
        )

        # Evaluate curve
        curve_points = evaluate_bezier_curve_fast(control_points, num_samples=bezier_samples)

        # Check area ratio (color balance)
        avg_x = np.mean(curve_points[:, 0])
        ratio = avg_x / size

        if min_ratio <= ratio <= max_ratio:
            # Generate texture using fast distance field
            texture_array = create_gradient_fast(
                curve_points, size, color1, color2, steepness, downsample_factor
            )

            if visualize:
                # Create debug version with curve overlay
                debug_img = visualize_curve_on_texture(
                    texture_array.copy(), control_points, curve_points
                )
                return Image.fromarray(texture_array), debug_img
            else:
                return Image.fromarray(texture_array)

    raise RuntimeError(f"Failed to generate texture after {max_attempts} attempts. "
                      f"Try adjusting TEXTURE_MIN_AREA_RATIO and TEXTURE_MAX_AREA_RATIO in config.py")


def visualize_curve_on_texture(texture, control_points, curve_points):
    """Draw curve on texture for debugging.

    Args:
        texture: RGB texture array
        control_points: Bezier control points
        curve_points: Evaluated curve points

    Returns:
        PIL.Image: Texture with curve overlay
    """
    img = Image.fromarray(texture)
    draw = ImageDraw.Draw(img)

    # Draw curve
    curve_coords = [(int(p[0]), int(p[1])) for p in curve_points[::config.TEXTURE_DEBUG_CURVE_SKIP]]
    draw.line(curve_coords, fill=(255, 255, 0), width=config.TEXTURE_DEBUG_CURVE_WIDTH)

    # Draw control points
    radius = config.TEXTURE_DEBUG_CONTROL_POINT_RADIUS
    for cp in control_points:
        x, y = int(cp[0]), int(cp[1])
        draw.ellipse(
            [x-radius, y-radius, x+radius, y+radius],
            fill=(0, 0, 255),
            outline=(255, 255, 255),
            width=2
        )

    return img


def generate_all_textures(output_dir=None, visualize=False):
    """Generate all texture variants."""
    if output_dir is None:
        output_dir = config.DATA_DIR / "objects" / "textures"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Ultra-Fast Texture Generator ===")
    print(f"Numba JIT: {'ENABLED' if NUMBA_AVAILABLE else 'DISABLED (install numba for 4x speedup)'}")
    print(f"Downsample factor: {config.TEXTURE_DOWNSAMPLE_FACTOR}x ({config.TEXTURE_DOWNSAMPLE_FACTOR**2}x speedup)")
    print(f"Bezier samples: {config.TEXTURE_BEZIER_SAMPLES}")

    # Red texture
    print("\n[1/3] Red texture...")
    red_texture = np.zeros((config.TEXTURE_SIZE, config.TEXTURE_SIZE, 3), dtype=np.uint8)
    red_texture[:, :] = config.TEXTURE_RED_COLOR
    red_path = output_dir / "red.png"
    Image.fromarray(red_texture).save(red_path)
    print(f"  Saved: {red_path}")

    # Mixed texture
    print("\n[2/3] Mixed texture (curved gradient)...")
    result = generate_curved_gradient_texture(
        color1=config.TEXTURE_RED_COLOR,
        color2=config.TEXTURE_GREEN_COLOR,
        complexity=config.TEXTURE_CURVE_COMPLEXITY,
        amplitude=config.TEXTURE_CURVE_AMPLITUDE,
        size=config.TEXTURE_SIZE,
        min_ratio=config.TEXTURE_MIN_AREA_RATIO,
        max_ratio=config.TEXTURE_MAX_AREA_RATIO,
        steepness=config.TEXTURE_GRADIENT_STEEPNESS,
        start_x_min=config.TEXTURE_CURVE_START_X_MIN,
        start_x_max=config.TEXTURE_CURVE_START_X_MAX,
        end_x_min=config.TEXTURE_CURVE_END_X_MIN,
        end_x_max=config.TEXTURE_CURVE_END_X_MAX,
        extension_top=config.TEXTURE_CURVE_EXTENSION_TOP,
        extension_bottom=config.TEXTURE_CURVE_EXTENSION_BOTTOM,
        bezier_samples=config.TEXTURE_BEZIER_SAMPLES,
        downsample_factor=config.TEXTURE_DOWNSAMPLE_FACTOR,
        max_attempts=config.TEXTURE_MAX_GENERATION_ATTEMPTS,
        visualize=visualize
    )

    if visualize:
        mixed_texture, debug_texture = result
        mixed_path = output_dir / "mixed.png"
        debug_path = output_dir / "mixed_debug.png"
        mixed_texture.save(mixed_path)
        debug_texture.save(debug_path)
        print(f"  Saved: {mixed_path}")
        print(f"  Debug: {debug_path}")
    else:
        mixed_texture = result
        mixed_path = output_dir / "mixed.png"
        mixed_texture.save(mixed_path)
        print(f"  Saved: {mixed_path}")

    # Green texture
    print("\n[3/3] Green texture...")
    green_texture = np.zeros((config.TEXTURE_SIZE, config.TEXTURE_SIZE, 3), dtype=np.uint8)
    green_texture[:, :] = config.TEXTURE_GREEN_COLOR
    green_path = output_dir / "green.png"
    Image.fromarray(green_texture).save(green_path)
    print(f"  Saved: {green_path}")

    print("\n=== Complete ===")
    return {"red": red_path, "mixed": mixed_path, "green": green_path}


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="Generate debug version with curve overlay")
    args = parser.parse_args()

    start_time = time.perf_counter()
    texture_paths = generate_all_textures(visualize=args.visualize)
    elapsed = time.perf_counter() - start_time

    print(f"\nTotal time: {elapsed:.2f}s")
