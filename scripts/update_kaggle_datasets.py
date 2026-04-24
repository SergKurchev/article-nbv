"""
Automatic Kaggle dataset version updater.

This script:
1. Checks if datasets exist on Kaggle
2. Automatically creates new version with updated data
3. Handles all 3 stages in one command
4. Shows progress and verification
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import subprocess
import os
import time
from datetime import datetime

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

    print(f"[OK] Kaggle credentials configured")

def check_dataset_exists(dataset_slug):
    """Check if dataset exists on Kaggle."""
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "status", dataset_slug],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except:
        return False

def get_dataset_info(dataset_slug):
    """Get dataset information from Kaggle."""
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "metadata", dataset_slug, "-p", "."],
            capture_output=True,
            text=True,
            cwd=Path.home()
        )

        if result.returncode == 0:
            metadata_file = Path.home() / "dataset-metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    content = f.read().strip()
                metadata_file.unlink()  # Clean up

                # Kaggle CLI returns double-encoded JSON (JSON string inside JSON)
                # First parse to get the string, then parse again to get the dict
                try:
                    info = json.loads(content)
                    # If it's a string, parse again
                    if isinstance(info, str):
                        info = json.loads(info)
                    return info
                except json.JSONDecodeError as e:
                    print(f"[WARNING] Could not parse metadata: {e}")
                    return None
            # If no file was created, try parsing stdout as JSON
            elif result.stdout.strip():
                try:
                    info = json.loads(result.stdout)
                    if isinstance(info, str):
                        info = json.loads(info)
                    return info
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"[WARNING] Error getting dataset info: {e}")
    return None

def update_dataset_version(dataset_dir, stage, username, version_notes=None):
    """Update existing dataset with new version."""
    print(f"\n{'='*60}")
    print(f"Updating Stage {stage} Dataset on Kaggle")
    print(f"{'='*60}")

    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        return False

    # Check metadata file exists
    metadata_file = dataset_dir / "dataset-metadata.json"
    if not metadata_file.exists():
        print(f"[ERROR] Metadata file not found: {metadata_file}")
        return False

    # Get dataset slug
    with open(metadata_file) as f:
        metadata = json.load(f)

    dataset_slug = metadata.get("id", f"{username}/nbv-stage{stage}-dataset")

    # Check if dataset exists
    exists = check_dataset_exists(dataset_slug)

    if not exists:
        print(f"[INFO] Dataset does not exist yet. Creating new dataset...")
        return create_new_dataset(dataset_dir, stage)

    # Get current version info
    print(f"[INFO] Checking current version...")
    info = get_dataset_info(dataset_slug)
    if info and isinstance(info, dict):
        current_version = info.get("datasetId", "unknown")
        print(f"[INFO] Current dataset ID: {current_version}")

    # Prepare version notes
    if version_notes is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        version_notes = f"Updated dataset - {timestamp}\n\nChanges:\n- Fixed stage-aware generation\n- Correct object counts per stage\n- Fixed visualization camera transforms"

    # Change to dataset directory
    original_dir = Path.cwd()
    os.chdir(dataset_dir)

    try:
        print("[INFO] Uploading new version...")
        result = subprocess.run(
            ["kaggle", "datasets", "version", "-p", ".", "-m", version_notes, "--dir-mode", "zip"],
            capture_output=True,
            text=True
        )

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode == 0:
            print(f"[OK] Successfully updated Stage {stage} dataset!")
            print(f"[INFO] URL: https://www.kaggle.com/datasets/{dataset_slug}")
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

def create_new_dataset(dataset_dir, stage):
    """Create new dataset on Kaggle."""
    print(f"[INFO] Creating new dataset for Stage {stage}...")

    original_dir = Path.cwd()
    os.chdir(dataset_dir)

    try:
        result = subprocess.run(
            ["kaggle", "datasets", "create", "-p", ".", "--dir-mode", "zip"],
            capture_output=True,
            text=True
        )

        print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode == 0:
            print(f"[OK] Successfully created Stage {stage} dataset!")
            return True
        else:
            print(f"[ERROR] Creation failed with return code {result.returncode}")
            return False

    except Exception as e:
        print(f"[ERROR] Creation failed: {e}")
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
        print(f"[OK] Kaggle CLI version: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("[ERROR] Kaggle CLI not found. Install with: pip install kaggle")
        return False

def count_samples(dataset_dir):
    """Count number of samples in dataset."""
    if not dataset_dir.exists():
        return 0
    samples = list(dataset_dir.glob("sample_*"))
    return len(samples)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Update NBV datasets on Kaggle")
    parser.add_argument("--username", type=str, default="sergeykurchev", help="Kaggle username")
    parser.add_argument("--key", type=str, default="fd9ae7ea316d408e492e260be6c3727e", help="Kaggle API key")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], help="Update specific stage only")
    parser.add_argument("--object-mode", type=str, default="primitives", help="Object mode (primitives/complex)")
    parser.add_argument("--notes", type=str, help="Custom version notes")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without actually uploading")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    print("="*60)
    print("NBV Kaggle Dataset Updater")
    print("="*60)

    # Setup credentials
    if not args.dry_run:
        setup_kaggle_credentials(args.username, args.key)

        # Verify Kaggle CLI
        if not verify_kaggle_setup():
            sys.exit(1)

    # Check datasets
    import config
    stages = [args.stage] if args.stage else [1, 2, 3]

    print(f"\n[INFO] Checking datasets...")
    for stage in stages:
        dataset_dir = config.BASE_DIR / "dataset" / args.object_mode / f"stage{stage}"
        sample_count = count_samples(dataset_dir)
        print(f"  Stage {stage}: {sample_count} samples")

    if args.dry_run:
        print("\n[DRY RUN] Would upload the above datasets to Kaggle")
        sys.exit(0)

    # Confirm upload
    print(f"\n[INFO] Ready to update {len(stages)} dataset(s) on Kaggle")
    if not args.yes:
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("[INFO] Upload cancelled")
            sys.exit(0)

    # Upload datasets
    results = {}
    for stage in stages:
        dataset_dir = config.BASE_DIR / "dataset" / args.object_mode / f"stage{stage}"
        success = update_dataset_version(dataset_dir, stage, args.username, args.notes)
        results[stage] = success

        # Small delay between uploads
        if stage != stages[-1]:
            time.sleep(2)

    # Summary
    print(f"\n{'='*60}")
    print("Update Summary")
    print(f"{'='*60}")
    for stage, success in results.items():
        status = "[OK] Success" if success else "[ERROR] Failed"
        url = f"https://www.kaggle.com/datasets/{args.username}/nbv-stage{stage}-dataset"
        print(f"Stage {stage}: {status}")
        if success:
            print(f"  URL: {url}")

    # Exit with error if any upload failed
    if not all(results.values()):
        sys.exit(1)

    print(f"\n[OK] All datasets updated successfully!")
