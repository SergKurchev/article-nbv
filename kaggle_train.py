import os
import subprocess
from pathlib import Path

# --- Configuration ---
# 1. Upload dataset.zip to Kaggle as a Dataset.
# 2. Update the path below to match your Kaggle dataset input name.
DATASET_ZIP = "/kaggle/input/nbv-multimodal-dataset/dataset.zip" 
REPO_URL = "https://github.com/REPLACE_WITH_YOUR_GITHUB_URL.git"

def run(cmd):
    print(f">> {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def setup():
    # Clone repository
    repo_name = Path(REPO_URL).stem
    if not os.path.exists(repo_name):
        run(f"git clone {REPO_URL}")
    os.chdir(repo_name)

    # Install dependencies
    # Kaggle has most ML libs, but we need pybullet, gymnasium, and others
    run("pip install pybullet gymnasium stable-baselines3 tqdm scikit-learn umap-learn")

    # Extract dataset
    if not os.path.exists("dataset"):
        print("Extracting dataset...")
        run(f"unzip -q {DATASET_ZIP} -d .")
    else:
        print("Dataset already exists.")

    # Train
    # Set IMAGE_CHANNELS=4 in config.py if not already set (it should be in the repo)
    print("Starting Training...")
    run("python src/vision/train_cnn.py")

if __name__ == "__main__":
    setup()
