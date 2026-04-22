import os
import shutil
from pathlib import Path

# --- KAGGLE SETUP ---
# On Kaggle, input datasets are in /kaggle/input/
# Your dataset name will be the folder name there.

DATASET_NAME = "nbv-rl-primitives-8obj" # Update this to your Kaggle dataset name!
INPUT_DIR = Path(f"/kaggle/input/{DATASET_NAME}")
OUTPUT_DIR = Path("/kaggle/working")

def setup_kaggle():
    print("Setting up Kaggle environment...")
    
    if not INPUT_DIR.exists():
        print(f"Error: Dataset {DATASET_NAME} not found in /kaggle/input/. Check your dataset name.")
        return False
        
    # On Kaggle, the project root should be current dir.
    # We assume you uploaded the source code or cloned the repo.
    print("Current Working Directory:", os.getcwd())
    
    # Check if src/ is present
    if not os.path.exists("src"):
        print("Error: 'src/' folder not found in current directory. Please upload your project files.")
        return False

    return True

def run_training():
    if not setup_kaggle():
        return

    # IMPORT CONFIG TO MODIFY PATHS LOCALLY FOR KAGGLE
    import config
    # Overwrite dataset path to point to Kaggle input
    config.DATASET_DIR = INPUT_DIR
    print(f"Overridden DATASET_DIR: {config.DATASET_DIR}")
    
    # Start training
    from src.vision.train_cnn import train_cnn
    print("\nStarting CNN Training on Kaggle...")
    train_cnn()
    
    # 4. ZIP RESULTS FOR EASY DOWNLOAD
    print("\nTraining Finished. Packing results...")
    shutil.make_archive("kaggle_results", 'zip', "weights")
    print("Results packed in /kaggle/working/kaggle_results.zip")

if __name__ == "__main__":
    run_training()
