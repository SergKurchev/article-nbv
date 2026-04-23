"""
Automated script to generate 1000-sample datasets for all stages and upload to Kaggle.

This script:
1. Generates textures (20 mixed variants)
2. Generates 1000-sample datasets for Stage 1, 2, and 3
3. Verifies data integrity
4. Uploads each dataset to Kaggle with proper metadata

Run in background mode to avoid blocking.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
import json
import subprocess
import shutil
from tqdm import tqdm

# Kaggle credentials
KAGGLE_USERNAME = "sergeykurchev"
KAGGLE_KEY = "fd9ae7ea316d408e492e260be6c3727e"

# Dataset configuration
TOTAL_SAMPLES = 1000
SAMPLES_PER_CLASS = TOTAL_SAMPLES // config.NUM_CLASSES


def setup_kaggle_credentials():
    """Setup Kaggle API credentials."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)

    kaggle_json = kaggle_dir / "kaggle.json"
    credentials = {
        "username": KAGGLE_USERNAME,
        "key": KAGGLE_KEY
    }

    with open(kaggle_json, 'w') as f:
        json.dump(credentials, f)

    # Set permissions (Unix-like systems)
    try:
        kaggle_json.chmod(0o600)
    except:
        pass

    print(f"Kaggle credentials configured at {kaggle_json}")


def generate_textures():
    """Generate 20 mixed texture variants."""
    print("\n" + "="*60)
    print("STEP 1: Generating Textures")
    print("="*60)

    from src.vision.texture_generator import generate_all_textures

    texture_dir = config.DATA_DIR / "objects" / "textures"
    texture_dir.mkdir(parents=True, exist_ok=True)

    # Check if textures already exist
    existing_mixed = list(texture_dir.glob("mixed_*.png"))
    if len(existing_mixed) >= config.TEXTURE_NUM_MIXED_VARIANTS:
        print(f"Found {len(existing_mixed)} mixed textures, skipping generation")
        return

    print(f"Generating {config.TEXTURE_NUM_MIXED_VARIANTS} mixed texture variants...")
    generate_all_textures(output_dir=texture_dir, visualize=False)
    print("Textures generated successfully")


def generate_stage_dataset(stage):
    """Generate dataset for a specific stage."""
    print(f"\n" + "="*60)
    print(f"STEP 2.{stage}: Generating Stage {stage} Dataset ({TOTAL_SAMPLES} samples)")
    print("="*60)

    # Update config temporarily
    original_stage = config.SCENE_STAGE
    original_samples = config.DATASET_SAMPLES_PER_CLASS

    config.SCENE_STAGE = stage
    config.DATASET_SAMPLES_PER_CLASS = SAMPLES_PER_CLASS

    # Set dataset directory
    dataset_dir = config.BASE_DIR / "dataset" / config.OBJECT_MODE / f"stage{stage}"
    config.DATASET_DIR = dataset_dir

    # Clear existing dataset if it exists
    if dataset_dir.exists():
        print(f"Removing existing dataset at {dataset_dir}")
        shutil.rmtree(dataset_dir)

    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Stage: {stage}")
    print(f"Samples per class: {SAMPLES_PER_CLASS}")
    print(f"Total samples: {TOTAL_SAMPLES}")
    print(f"Output directory: {dataset_dir}")

    # Generate dataset
    from src.vision.dataset_stage import generate_stage_dataset
    generate_stage_dataset()

    # Restore config
    config.SCENE_STAGE = original_stage
    config.DATASET_SAMPLES_PER_CLASS = original_samples
    config.DATASET_DIR = config.BASE_DIR / "dataset" / config.OBJECT_MODE

    return dataset_dir


def verify_dataset(dataset_dir, stage):
    """Verify dataset integrity."""
    print(f"\n" + "="*60)
    print(f"STEP 3.{stage}: Verifying Stage {stage} Dataset")
    print("="*60)

    from scripts.prepare_stage_datasets import verify_dataset as verify_fn
    is_valid = verify_fn(dataset_dir, f"Stage {stage}")

    if not is_valid:
        raise RuntimeError(f"Stage {stage} dataset verification failed!")

    return is_valid


def create_kaggle_metadata(dataset_dir, stage):
    """Create Kaggle dataset metadata."""
    print(f"\n" + "="*60)
    print(f"STEP 4.{stage}: Creating Kaggle Metadata for Stage {stage}")
    print("="*60)

    stage_names = {
        1: "Single Object",
        2: "Multi-Object",
        3: "Multi-Object with Obstacles"
    }

    metadata = {
        "title": f"StrawPick NBV Stage {stage} Dataset - {stage_names[stage]}",
        "id": f"{KAGGLE_USERNAME}/strawpick-nbv-stage{stage}-dataset",
        "licenses": [{"name": "CC0-1.0"}],
        "description": f"""
# StrawPick NBV Stage {stage} Dataset

Next-Best-View (NBV) training dataset for active object classification with robotic manipulation.

## Stage {stage}: {stage_names[stage]}

### Scene Configuration
- **Objects per scene**: {'1 (fixed position)' if stage == 1 else '2-10 (random placement)'}
- **Obstacles**: {'None' if stage <= 2 else '1-5 (random panels)'}
- **Object classes**: 8 primitive shapes (cube, sphere, cylinder, cone, torus, capsule, ellipsoid, pyramid)
- **Textures**: 3 types (red, mixed gradient with 20 variants, green)
- **Total samples**: {TOTAL_SAMPLES}

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

### Texture System

Objects are textured with one of three types:
- **Red**: Solid red texture
- **Mixed**: Gradient from red to green (20 pre-generated variants randomly selected)
- **Green**: Solid green texture

The mixed textures use Bezier curves with random seeds for reproducibility.

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
- **Vision models**: MultiModalNet, LightweightODIN (ODIN-inspired with Bayesian uncertainty)
- **RL agents**: SAC-based Next-Best-View policy

### Progressive Training

Train models progressively through stages:
1. **Stage 1**: Learn basic object recognition
2. **Stage 2**: Handle multiple objects and occlusions
3. **Stage 3**: Navigate around obstacles

### Citation

If you use this dataset, please cite:
```
StrawPick NBV Dataset - Skoltech Reinforcement Learning Project
Next-Best-View with Obstacles and Robot
```

### Project Repository

https://github.com/yourusername/NBV_with_obstacles_and_robot
""",
        "keywords": ["computer-vision", "3d", "robotics", "active-learning", "next-best-view", "strawpick", "nbv"]
    }

    metadata_file = dataset_dir / "dataset-metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata created: {metadata_file}")

    # Create README
    readme_file = dataset_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(f"# StrawPick NBV Stage {stage} Dataset\n\n")
        f.write(f"See dataset-metadata.json for full description.\n\n")
        f.write(f"## Quick Stats\n\n")
        f.write(f"- Stage: {stage}\n")
        f.write(f"- Scene type: {stage_names[stage]}\n")
        f.write(f"- Total samples: {len(list(dataset_dir.glob('sample_*')))}\n")
        f.write(f"- Object classes: {config.NUM_CLASSES}\n")
        f.write(f"- Texture variants: 20 mixed + 1 red + 1 green\n")

    print(f"README created: {readme_file}")


def upload_to_kaggle(dataset_dir, stage, create_new=True):
    """Upload dataset to Kaggle."""
    print(f"\n" + "="*60)
    print(f"STEP 5.{stage}: Uploading Stage {stage} Dataset to Kaggle")
    print("="*60)

    dataset_slug = f"strawpick-nbv-stage{stage}-dataset"

    if create_new:
        # Create new dataset
        print(f"Creating new Kaggle dataset: {KAGGLE_USERNAME}/{dataset_slug}")
        cmd = [
            "kaggle", "datasets", "create",
            "-p", str(dataset_dir),
            "-r", "zip"
        ]
    else:
        # Update existing dataset
        print(f"Updating Kaggle dataset: {KAGGLE_USERNAME}/{dataset_slug}")
        cmd = [
            "kaggle", "datasets", "version",
            "-p", str(dataset_dir),
            "-m", f"Updated with {TOTAL_SAMPLES} samples",
            "-r", "zip"
        ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"Successfully uploaded Stage {stage} dataset!")
        print(f"URL: https://www.kaggle.com/datasets/{KAGGLE_USERNAME}/{dataset_slug}")
    except subprocess.CalledProcessError as e:
        print(f"Error uploading dataset: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        if create_new:
            print("\nTrying to update existing dataset instead...")
            upload_to_kaggle(dataset_dir, stage, create_new=False)


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("KAGGLE DATASET GENERATION AND UPLOAD")
    print("="*60)
    print(f"Total samples per stage: {TOTAL_SAMPLES}")
    print(f"Samples per class: {SAMPLES_PER_CLASS}")
    print(f"Number of classes: {config.NUM_CLASSES}")
    print(f"Kaggle username: {KAGGLE_USERNAME}")
    print("="*60)

    # Setup Kaggle credentials
    setup_kaggle_credentials()

    # Generate textures
    generate_textures()

    # Process each stage
    for stage in [1, 2, 3]:
        try:
            # Generate dataset
            dataset_dir = generate_stage_dataset(stage)

            # Verify dataset
            verify_dataset(dataset_dir, stage)

            # Create Kaggle metadata
            create_kaggle_metadata(dataset_dir, stage)

            # Upload to Kaggle
            upload_to_kaggle(dataset_dir, stage, create_new=True)

            print(f"\n✓ Stage {stage} completed successfully!")

        except Exception as e:
            print(f"\n✗ Error processing Stage {stage}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*60)
    print("ALL STAGES COMPLETED")
    print("="*60)
    print("\nDatasets uploaded to Kaggle:")
    for stage in [1, 2, 3]:
        print(f"  Stage {stage}: https://www.kaggle.com/datasets/{KAGGLE_USERNAME}/strawpick-nbv-stage{stage}-dataset")


if __name__ == "__main__":
    main()
