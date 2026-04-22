"""
Upload NBV stage datasets to Kaggle.

This script:
1. Configures Kaggle API credentials
2. Uploads each stage dataset separately
3. Verifies upload success
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import subprocess
import os

def setup_kaggle_credentials(username, key):
    """Setup Kaggle API credentials."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)

    credentials_file = kaggle_dir / "kaggle.json"

    credentials = {
        "username": username,
        "key": key
    }

    with open(credentials_file, 'w') as f:
        json.dump(credentials, f)

    # Set permissions (Unix-like systems)
    if os.name != 'nt':
        os.chmod(credentials_file, 0o600)

    print(f"[OK] Kaggle credentials configured at {credentials_file}")

def upload_dataset_to_kaggle(dataset_dir, stage, create_new=True):
    """Upload dataset to Kaggle."""
    print(f"\n{'='*60}")
    print(f"Uploading Stage {stage} Dataset to Kaggle")
    print(f"{'='*60}")

    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        return False

    # Check metadata file exists
    metadata_file = dataset_dir / "dataset-metadata.json"
    if not metadata_file.exists():
        print(f"[ERROR] Metadata file not found: {metadata_file}")
        return False

    # Change to dataset directory
    original_dir = Path.cwd()
    os.chdir(dataset_dir)

    try:
        if create_new:
            # Create new dataset
            print("Creating new dataset on Kaggle...")
            result = subprocess.run(
                ["kaggle", "datasets", "create", "-p", ".", "--dir-mode", "zip"],
                capture_output=True,
                text=True
            )
        else:
            # Update existing dataset
            print("Updating existing dataset on Kaggle...")
            result = subprocess.run(
                ["kaggle", "datasets", "version", "-p", ".", "-m", f"Updated Stage {stage} dataset", "--dir-mode", "zip"],
                capture_output=True,
                text=True
            )

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode == 0:
            print(f"[OK] Successfully uploaded Stage {stage} dataset!")
            return True
        else:
            print(f"[ERROR] Upload failed with return code {result.returncode}")
            return False

    except FileNotFoundError:
        print("[ERROR] Kaggle CLI not found. Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        return False
    finally:
        os.chdir(original_dir)

def verify_kaggle_setup():
    """Verify Kaggle CLI is installed and configured."""
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True,
            text=True
        )
        print(f"Kaggle CLI version: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("[ERROR] Kaggle CLI not found. Install with: pip install kaggle")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str, default="sergeykurchev", help="Kaggle username")
    parser.add_argument("--key", type=str, default="fd9ae7ea316d408e492e260be6c3727e", help="Kaggle API key")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], help="Upload specific stage only")
    parser.add_argument("--update", action="store_true", help="Update existing dataset instead of creating new")
    parser.add_argument("--object-mode", type=str, default="primitives", help="Object mode (primitives/complex)")
    args = parser.parse_args()

    # Setup credentials
    setup_kaggle_credentials(args.username, args.key)

    # Verify Kaggle CLI
    if not verify_kaggle_setup():
        sys.exit(1)

    # Upload datasets
    import config
    stages = [args.stage] if args.stage else [1, 2, 3]

    results = {}
    for stage in stages:
        dataset_dir = config.BASE_DIR / "dataset" / args.object_mode / f"stage{stage}"
        success = upload_dataset_to_kaggle(dataset_dir, stage, create_new=not args.update)
        results[stage] = success

    # Summary
    print(f"\n{'='*60}")
    print("Upload Summary")
    print(f"{'='*60}")
    for stage, success in results.items():
        status = "[OK] Success" if success else "[ERROR] Failed"
        print(f"Stage {stage}: {status}")

    # Exit with error if any upload failed
    if not all(results.values()):
        sys.exit(1)
