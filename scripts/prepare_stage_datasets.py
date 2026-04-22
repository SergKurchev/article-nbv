"""
Prepare separate datasets for each training stage and verify data integrity.

This script:
1. Generates datasets for Stage 1, 2, and 3
2. Verifies data integrity for each stage
3. Creates metadata files for Kaggle upload
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil

def verify_sample(sample_dir):
    """Verify a single sample has all required files and valid data."""
    issues = []

    # Check directories exist
    required_dirs = ['rgb', 'depth', 'masks', 'labels']
    for d in required_dirs:
        if not (sample_dir / d).exists():
            issues.append(f"Missing directory: {d}")
            return issues

    # Check metadata files
    if not (sample_dir / "color_map.json").exists():
        issues.append("Missing color_map.json")
    if not (sample_dir / "cameras.json").exists():
        issues.append("Missing cameras.json")

    # Load and verify cameras.json
    try:
        with open(sample_dir / "cameras.json") as f:
            cameras = json.load(f)
        if len(cameras) == 0:
            issues.append("No camera views in cameras.json")
    except Exception as e:
        issues.append(f"Invalid cameras.json: {e}")
        return issues

    # Verify each view has matching files
    for view_id in cameras.keys():
        rgb_path = sample_dir / "rgb" / f"{view_id}.png"
        depth_path = sample_dir / "depth" / f"{view_id}.npy"
        mask_path = sample_dir / "masks" / f"{view_id}.png"
        label_path = sample_dir / "labels" / f"{view_id}.txt"

        if not rgb_path.exists():
            issues.append(f"Missing RGB: {view_id}.png")
        if not depth_path.exists():
            issues.append(f"Missing depth: {view_id}.npy")
        if not mask_path.exists():
            issues.append(f"Missing mask: {view_id}.png")
        if not label_path.exists():
            issues.append(f"Missing label: {view_id}.txt")

        # Verify data can be loaded
        try:
            if rgb_path.exists():
                img = Image.open(rgb_path)
                if img.size != (config.IMAGE_SIZE, config.IMAGE_SIZE):
                    issues.append(f"Wrong RGB size for {view_id}: {img.size}")
        except Exception as e:
            issues.append(f"Cannot load RGB {view_id}: {e}")

        try:
            if depth_path.exists():
                depth = np.load(depth_path)
                if depth.shape != (config.IMAGE_SIZE, config.IMAGE_SIZE):
                    issues.append(f"Wrong depth shape for {view_id}: {depth.shape}")
        except Exception as e:
            issues.append(f"Cannot load depth {view_id}: {e}")

    return issues

def verify_dataset(dataset_dir, stage_name):
    """Verify entire dataset integrity."""
    print(f"\n{'='*60}")
    print(f"Verifying {stage_name} dataset at {dataset_dir}")
    print(f"{'='*60}")

    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory does not exist!")
        return False

    samples = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")])

    if len(samples) == 0:
        print(f"[ERROR] No samples found!")
        return False

    print(f"Found {len(samples)} samples")

    # Verify random samples
    import random
    samples_to_check = random.sample(samples, min(10, len(samples)))

    all_valid = True
    for sample_dir in tqdm(samples_to_check, desc="Verifying samples"):
        issues = verify_sample(sample_dir)
        if issues:
            print(f"\n[ERROR] Issues in {sample_dir.name}:")
            for issue in issues:
                print(f"  - {issue}")
            all_valid = False

    if all_valid:
        print(f"[OK] All checked samples are valid!")

    # Collect statistics
    total_views = 0
    for sample_dir in samples[:100]:  # Check first 100 for stats
        cameras_file = sample_dir / "cameras.json"
        if cameras_file.exists():
            with open(cameras_file) as f:
                cameras = json.load(f)
                total_views += len(cameras)

    avg_views = total_views / min(100, len(samples))
    print(f"\nStatistics:")
    print(f"  Total samples: {len(samples)}")
    print(f"  Average views per sample: {avg_views:.1f}")
    print(f"  Estimated total views: {int(len(samples) * avg_views)}")

    return all_valid

def create_dataset_metadata(dataset_dir, stage, output_file):
    """Create metadata file for Kaggle dataset."""
    metadata = {
        "title": f"NBV Stage {stage} Dataset - {'Single Object' if stage == 1 else 'Multi-Object' if stage == 2 else 'Multi-Object + Obstacles'}",
        "id": f"sergeykurchev/nbv-stage{stage}-dataset",
        "licenses": [{"name": "CC0-1.0"}],
        "description": f"""
# NBV Stage {stage} Dataset

Next-Best-View (NBV) training dataset for active object classification.

## Stage {stage}: {'Single Object (Baseline)' if stage == 1 else 'Multiple Objects' if stage == 2 else 'Multiple Objects + Obstacles'}

### Scene Configuration
- **Objects per scene**: {'1 (fixed position)' if stage == 1 else '2-10 (random placement)' if stage == 2 else '2-10 (random placement)'}
- **Obstacles**: {'0' if stage <= 2 else '1-5 (random panels)'}
- **Object classes**: 8 primitive shapes (cube, sphere, cylinder, cone, torus, capsule, ellipsoid, pyramid)
- **Textures**: 3 types (red, mixed gradient, green)

### Data Format

Each sample is a directory `sample_XXXXX/` containing:

- `rgb/` - RGB images (224x224 PNG)
- `depth/` - Depth maps (224x224 NPY, float32, meters)
- `masks/` - Instance segmentation masks (224x224 PNG, colored)
- `labels/` - YOLO format bounding boxes (TXT)
- `cameras.json` - Camera parameters (position, rotation, intrinsics)
- `color_map.json` - Instance ID to class mapping

### Camera Views

- **Views per sample**: ~20 (spherical orbit around objects)
- **Camera distance**: 0.3-0.6 meters
- **Elevation angle**: 0.1-1.37 radians
- **Image size**: 224x224 pixels

### Usage

```python
import json
import numpy as np
from PIL import Image

# Load a sample
sample_dir = "sample_00000"

# Load RGB
rgb = Image.open(sample_dir + "/rgb/00000.png")

# Load depth
depth = np.load(sample_dir + "/depth/00000.npy")

# Load metadata
with open(sample_dir + "/cameras.json") as f:
    cameras = json.load(f)

with open(sample_dir + "/color_map.json") as f:
    color_map = json.load(f)
```

### Training Models

This dataset is designed for training:
- **Vision models**: MultiModalNet, LightweightODIN (ODIN-inspired)
- **RL agents**: SAC-based Next-Best-View policy

See project repository for training scripts.

### Citation

If you use this dataset, please cite:
```
NBV with Obstacles and Robot - Skoltech Reinforcement Learning Project
```
""",
        "keywords": ["computer-vision", "3d", "robotics", "active-learning", "next-best-view"]
    }

    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Created metadata: {output_file}")

def prepare_stage_dataset(stage, num_samples=100):
    """Generate and verify dataset for a specific stage."""
    print(f"\n{'='*60}")
    print(f"Preparing Stage {stage} Dataset")
    print(f"{'='*60}")

    # Update config
    original_stage = config.SCENE_STAGE
    original_samples = config.DATASET_SAMPLES_PER_CLASS

    config.SCENE_STAGE = stage
    config.DATASET_SAMPLES_PER_CLASS = num_samples // config.NUM_CLASSES

    # Set dataset directory
    dataset_dir = config.BASE_DIR / "dataset" / config.OBJECT_MODE / f"stage{stage}"
    config.DATASET_DIR = dataset_dir

    print(f"Stage: {stage}")
    print(f"Samples per class: {config.DATASET_SAMPLES_PER_CLASS}")
    print(f"Total samples: {config.DATASET_SAMPLES_PER_CLASS * config.NUM_CLASSES}")
    print(f"Output directory: {dataset_dir}")

    # Generate dataset
    from src.vision.dataset_stage import generate_stage_dataset
    generate_stage_dataset()

    # Verify dataset
    is_valid = verify_dataset(dataset_dir, f"Stage {stage}")

    # Create metadata for Kaggle
    metadata_file = dataset_dir / "dataset-metadata.json"
    create_dataset_metadata(dataset_dir, stage, metadata_file)

    # Create README
    readme_file = dataset_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(f"# NBV Stage {stage} Dataset\n\n")
        f.write(f"See dataset-metadata.json for full description.\n\n")
        f.write(f"## Quick Stats\n\n")
        f.write(f"- Stage: {stage}\n")
        f.write(f"- Scene type: {'Single object' if stage == 1 else 'Multi-object' if stage == 2 else 'Multi-object + obstacles'}\n")
        f.write(f"- Total samples: {len(list(dataset_dir.glob('sample_*')))}\n")
        f.write(f"- Object classes: {config.NUM_CLASSES}\n")

    # Restore config
    config.SCENE_STAGE = original_stage
    config.DATASET_SAMPLES_PER_CLASS = original_samples
    config.DATASET_DIR = config.BASE_DIR / "dataset" / config.OBJECT_MODE

    return is_valid, dataset_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], help="Generate specific stage only")
    parser.add_argument("--samples", type=int, default=100, help="Total samples to generate (default: 100)")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing datasets")
    args = parser.parse_args()

    if args.verify_only:
        # Verify existing datasets
        for stage in [1, 2, 3]:
            dataset_dir = config.BASE_DIR / "dataset" / config.OBJECT_MODE / f"stage{stage}"
            if dataset_dir.exists():
                verify_dataset(dataset_dir, f"Stage {stage}")
    else:
        # Generate datasets
        stages = [args.stage] if args.stage else [1, 2, 3]

        results = {}
        for stage in stages:
            is_valid, dataset_dir = prepare_stage_dataset(stage, num_samples=args.samples)
            results[stage] = (is_valid, dataset_dir)

        # Summary
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        for stage, (is_valid, dataset_dir) in results.items():
            status = "[OK] Valid" if is_valid else "[ERROR] Invalid"
            print(f"Stage {stage}: {status} - {dataset_dir}")
