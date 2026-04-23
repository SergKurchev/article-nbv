"""
Push notebook to Kaggle and start training.

This script:
1. Creates Kaggle kernel metadata
2. Pushes notebook to Kaggle
3. Monitors execution status
"""

import json
import subprocess
import time
from pathlib import Path

# Configuration
NOTEBOOK_PATH = Path("notebooks/kaggle_training.ipynb")
KERNEL_SLUG = "nbv-training-full"
TITLE = "NBV Training - PointNet + RL"

# Dataset inputs
DATASETS = [
    "sergeykurchev/nbv-stage3-dataset"
]

def create_kernel_metadata():
    """Create kernel-metadata.json for Kaggle."""
    metadata = {
        "id": f"sergeykurchev/{KERNEL_SLUG}",
        "title": TITLE,
        "code_file": "kaggle_training.ipynb",  # Relative to metadata file
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": DATASETS,
        "competition_sources": [],
        "kernel_sources": []
    }

    metadata_path = NOTEBOOK_PATH.parent / "kernel-metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Created metadata: {metadata_path}")
    return metadata_path

def push_kernel():
    """Push kernel to Kaggle."""
    print("\nPushing kernel to Kaggle...")

    result = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(NOTEBOOK_PATH.parent)],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to push kernel: {result.stderr}")

    print(f"\n✓ Kernel pushed successfully!")
    print(f"URL: https://www.kaggle.com/code/sergeykurchev/{KERNEL_SLUG}")

def check_status():
    """Check kernel execution status."""
    print("\nChecking kernel status...")

    result = subprocess.run(
        ["kaggle", "kernels", "status", f"sergeykurchev/{KERNEL_SLUG}"],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    return result.stdout

def main():
    print("="*60)
    print("Kaggle Kernel Push Script")
    print("="*60)

    # Create metadata
    metadata_path = create_kernel_metadata()

    # Push kernel
    push_kernel()

    # Wait a bit for Kaggle to process
    print("\nWaiting for Kaggle to process...")
    time.sleep(5)

    # Check status
    status = check_status()

    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print(f"1. Go to: https://www.kaggle.com/code/sergeykurchev/{KERNEL_SLUG}")
    print("2. Click 'Run All' to start training")
    print("3. Monitor progress in the notebook")
    print("4. Download 'kaggle_results.zip' when complete")
    print("="*60)

if __name__ == "__main__":
    main()
