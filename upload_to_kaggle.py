import os
import shutil
import json
import subprocess
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))
import config

def upload_dataset():
    dataset_name = f"nbv-rl-{config.OBJECT_MODE}-{config.NUM_CLASSES}obj"
    dataset_title = f"NBV RL Dataset ({config.OBJECT_MODE}, {config.NUM_CLASSES} classes)"
    
    # Paths
    dataset_path = config.DATASET_DIR
    tmp_dir = Path("tmp_kaggle_upload")
    zip_name = "dataset"
    
    if not dataset_path.exists():
        print(f"Error: Dataset directory {dataset_path} not found.")
        return

    print(f"Preparing dataset: {dataset_name}...")
    
    # 1. Create clean temp directory for upload
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()
    
    # 2. Archive the dataset
    print(f"Zipping dataset from {dataset_path} (this may take a minute)...")
    shutil.make_archive(str(tmp_dir / zip_name), 'zip', dataset_path)
    
    # 3. Create dataset-metadata.json
    # Note: User needs to change 'your-username' later or we try to get it
    metadata = {
        "title": dataset_title,
        "id": f"{{username}}/{dataset_name}",
        "licenses": [{"name": "CC0-1.0"}]
    }
    
    with open(tmp_dir / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("\nDataset package ready in 'tmp_kaggle_upload/'.")
    print("Checking for Kaggle CLI...")
    
    try:
        # Try to get username from kaggle config
        username = subprocess.check_output(["kaggle", "config", "view"], stderr=subprocess.STDOUT).decode()
        if "username: " in username:
            username = username.split("username: ")[1].split("\n")[0].strip()
            # Update metadata with real username
            metadata["id"] = f"{username}/{dataset_name}"
            with open(tmp_dir / "dataset-metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)
            print(f"Detected Kaggle username: {username}")
        else:
            print("Warning: Could not detect Kaggle username. Please edit 'tmp_kaggle_upload/dataset-metadata.json'.")
    except Exception:
        print("Warning: Kaggle CLI not found or not configured. You can upload manually or install it: pip install kaggle")
        print("Manual upload: Zip 'dataset/' and upload to Kaggle as a new dataset.")
        return

    print(f"\nRunning: kaggle datasets create -p {tmp_dir}")
    print("If it already exists, use: kaggle datasets version -p tmp_kaggle_upload -m 'Update dataset'")
    
    # We don't run it automatically to let user confirm metadata
    print("\nNext Steps:")
    print(f"1. cd {tmp_dir}")
    print("2. kaggle datasets create -p . (if first time)")
    print("3. kaggle datasets version -p . -m 'Update' (if exists)")

if __name__ == "__main__":
    upload_dataset()
