# %% [markdown]
# # NBV Project - Full Training Pipeline on Kaggle
#
# This notebook trains:
# 1. **PointNet** - Point cloud classification model
# 2. **RL Agent** - Reinforcement learning agent for Next Best View selection
#
# All training runs in `uv` virtual environment with automatic CSV logging for easy download.

# %% [markdown]
# ## Configuration

# %%
import os
import sys
import subprocess
import shutil
from pathlib import Path
import time

# Kaggle-specific paths
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
IS_KAGGLE = KAGGLE_INPUT.exists()

# Dataset configuration
DATASET_STAGE = 3  # Use Stage 3 (most complex: objects + obstacles)
DATASET_SLUG = f"sergeykurchev/nbv-stage{DATASET_STAGE}-dataset"

# Training configuration
POINTNET_EPOCHS = 50
RL_TIMESTEPS = 50000  # Reduced from 500k to fit Kaggle time limits
OBJECT_MODE = "primitives"

# Repository
REPO_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO.git"  # UPDATE THIS!

print(f"Running on Kaggle: {IS_KAGGLE}")
print(f"Dataset: Stage {DATASET_STAGE}")
print(f"PointNet epochs: {POINTNET_EPOCHS}")
print(f"RL timesteps: {RL_TIMESTEPS}")

# %% [markdown]
# ## 1. Environment Setup

# %%
def run_command(cmd, description="", check=True):
    """Run shell command with progress indicator."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")

    return result

# Install uv if not present
print("Installing uv package manager...")
run_command("pip install uv", "Installing uv")

# Clone repository (or use uploaded code)
if IS_KAGGLE:
    print("\nSetting up repository...")
    if not Path("NBV_with_obstacles_and_robot").exists():
        # Try to clone from GitHub
        try:
            run_command(f"git clone {REPO_URL} NBV_with_obstacles_and_robot", "Cloning repository")
        except:
            print("Warning: Could not clone repository. Make sure code is uploaded to Kaggle.")
            print("You can upload the entire project folder as a Kaggle dataset.")

    os.chdir("NBV_with_obstacles_and_robot")
else:
    print("Running locally - assuming we're already in project directory")

print(f"\nCurrent directory: {os.getcwd()}")
print(f"Files: {os.listdir('.')[:10]}")

# %% [markdown]
# ## 2. Download Dataset (Stage 3 Only)

# %%
if IS_KAGGLE:
    print(f"\nDownloading Stage {DATASET_STAGE} dataset from Kaggle...")

    # Check if dataset is already available as input
    dataset_input_path = KAGGLE_INPUT / f"nbv-stage{DATASET_STAGE}-dataset"

    if dataset_input_path.exists():
        print(f"Dataset found in Kaggle inputs: {dataset_input_path}")

        # Create symlink or copy to expected location
        dataset_target = Path(f"dataset/{OBJECT_MODE}/stage{DATASET_STAGE}")
        dataset_target.parent.mkdir(parents=True, exist_ok=True)

        if not dataset_target.exists():
            print(f"Copying dataset to {dataset_target}...")
            shutil.copytree(dataset_input_path, dataset_target)
            print(f"Dataset copied successfully")
    else:
        print(f"Dataset not found in inputs. Downloading via Kaggle API...")
        run_command(f"kaggle datasets download -d {DATASET_SLUG}", "Downloading dataset")

        # Unzip
        zip_file = f"nbv-stage{DATASET_STAGE}-dataset.zip"
        if Path(zip_file).exists():
            run_command(f"unzip -q {zip_file} -d dataset/{OBJECT_MODE}/stage{DATASET_STAGE}/", "Extracting dataset")
            Path(zip_file).unlink()  # Clean up zip

    # Verify dataset
    dataset_path = Path(f"dataset/{OBJECT_MODE}/stage{DATASET_STAGE}")
    if dataset_path.exists():
        samples = list(dataset_path.glob("sample_*"))
        print(f"\n✓ Dataset ready: {len(samples)} samples found")
    else:
        raise RuntimeError(f"Dataset not found at {dataset_path}")
else:
    print("Running locally - assuming dataset already exists")
    dataset_path = Path(f"dataset/{OBJECT_MODE}/stage{DATASET_STAGE}")
    if dataset_path.exists():
        samples = list(dataset_path.glob("sample_*"))
        print(f"✓ Dataset found: {len(samples)} samples")

# %% [markdown]
# ## 3. Generate PointNet Dataset
#
# Convert RGB-D images to point clouds for PointNet training.

# %%
print("\nGenerating PointNet dataset from RGB-D images...")
print("This will convert images to 3D point clouds...")

# Update config to use only Stage 3
config_override = f"""
import sys
sys.path.insert(0, '.')
import config
config.SCENE_STAGE = {DATASET_STAGE}
"""

# Run dataset generation
run_command(
    f"uv run python src/vision/dataset_pointnet.py",
    "Generating PointNet dataset"
)

# Verify PointNet dataset
pointnet_dataset_path = Path(f"dataset_pointnet/{OBJECT_MODE}")
if pointnet_dataset_path.exists():
    samples = list(pointnet_dataset_path.glob("sample_*"))
    print(f"\n✓ PointNet dataset ready: {len(samples)} samples")
else:
    raise RuntimeError(f"PointNet dataset not found at {pointnet_dataset_path}")

# %% [markdown]
# ## 4. Train PointNet Model
#
# Train point cloud classification model.

# %%
print("\nTraining PointNet model...")
print(f"Epochs: {POINTNET_EPOCHS}")
print(f"This will take approximately 1-2 hours on GPU...")

# Override config for training
with open("config_override.py", "w") as f:
    f.write(f"""
# Kaggle training overrides
POINTNET_EPOCHS = {POINTNET_EPOCHS}
SCENE_STAGE = {DATASET_STAGE}
""")

# Run training
start_time = time.time()
run_command(
    "uv run python src/vision/train_pointnet.py",
    "Training PointNet"
)
elapsed = time.time() - start_time

print(f"\n✓ PointNet training completed in {elapsed/60:.1f} minutes")

# Check for saved weights
weights_dir = Path("weights") / OBJECT_MODE
if (weights_dir / "pointnet_best.pt").exists():
    print(f"✓ Best model saved: {weights_dir / 'pointnet_best.pt'}")
if (weights_dir / "pointnet_metrics.csv").exists():
    print(f"✓ Training metrics saved: {weights_dir / 'pointnet_metrics.csv'}")

# %% [markdown]
# ## 5. Train RL Agent (Optional)
#
# Train reinforcement learning agent with reduced timesteps to fit Kaggle limits.

# %%
print("\nTraining RL agent...")
print(f"Timesteps: {RL_TIMESTEPS} (reduced for Kaggle)")
print(f"This will take approximately 2-4 hours...")

# Update config for RL training
with open("config_override.py", "a") as f:
    f.write(f"""
TOTAL_TIMESTEPS = {RL_TIMESTEPS}
CNN_ARCHITECTURE = "PointNet"
CNN_LOAD_MODE = "best"
""")

# Run RL training with headless mode
start_time = time.time()
try:
    run_command(
        "uv run python train.py --mode train --no_arm",
        "Training RL Agent",
        check=False  # Don't fail if timeout
    )
    elapsed = time.time() - start_time
    print(f"\n✓ RL training completed in {elapsed/60:.1f} minutes")
except Exception as e:
    print(f"\nWarning: RL training interrupted: {e}")
    print("Partial results will still be saved")

# Check for RL results
runs_dir = Path("runs")
if runs_dir.exists():
    latest_run = max(runs_dir.glob("rl_train_*"), key=lambda p: p.stat().st_mtime, default=None)
    if latest_run:
        print(f"✓ RL run directory: {latest_run.name}")
        if (latest_run / "rl_metrics.csv").exists():
            print(f"✓ RL metrics saved: {latest_run / 'rl_metrics.csv'}")
        if (latest_run / "best_policy.zip").exists():
            print(f"✓ Best policy saved: {latest_run / 'best_policy.zip'}")

# %% [markdown]
# ## 6. Package Results for Download

# %%
print("\nPackaging results...")

# Create results directory
results_dir = Path("kaggle_results")
results_dir.mkdir(exist_ok=True)

# Copy weights
if (weights_dir).exists():
    shutil.copytree(weights_dir, results_dir / "weights", dirs_exist_ok=True)
    print(f"✓ Copied weights to {results_dir / 'weights'}")

# Copy RL run (latest)
if runs_dir.exists():
    latest_run = max(runs_dir.glob("rl_train_*"), key=lambda p: p.stat().st_mtime, default=None)
    if latest_run:
        shutil.copytree(latest_run, results_dir / "rl_run", dirs_exist_ok=True)
        print(f"✓ Copied RL run to {results_dir / 'rl_run'}")

# Collect all CSV files
csv_files = []
csv_files.extend(weights_dir.glob("*.csv"))
if latest_run:
    csv_files.extend(latest_run.glob("*.csv"))

# Copy CSVs to results
for csv_file in csv_files:
    shutil.copy(csv_file, results_dir / csv_file.name)
    print(f"✓ Copied {csv_file.name}")

# Create summary
summary_path = results_dir / "SUMMARY.txt"
with open(summary_path, "w") as f:
    f.write("NBV Training Results\n")
    f.write("="*60 + "\n\n")
    f.write(f"Dataset: Stage {DATASET_STAGE}\n")
    f.write(f"PointNet epochs: {POINTNET_EPOCHS}\n")
    f.write(f"RL timesteps: {RL_TIMESTEPS}\n\n")
    f.write("Files:\n")
    for item in sorted(results_dir.rglob("*")):
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            f.write(f"  {item.relative_to(results_dir)} ({size_mb:.2f} MB)\n")

print(f"\n✓ Summary saved: {summary_path}")

# Create zip archive
print("\nCreating zip archive...")
shutil.make_archive("kaggle_results", 'zip', results_dir)
zip_size = Path("kaggle_results.zip").stat().st_size / (1024 * 1024)
print(f"✓ Results packaged: kaggle_results.zip ({zip_size:.2f} MB)")

# %% [markdown]
# ## 7. Training Complete!
#
# Download `kaggle_results.zip` from the output panel.
#
# ### Contents:
# - `weights/pointnet_best.pt` - Best PointNet model
# - `weights/pointnet_metrics.csv` - PointNet training metrics
# - `rl_run/best_policy.zip` - Best RL policy
# - `rl_run/rl_metrics.csv` - RL training metrics
# - `rl_run/train_cnn_probs.csv` - CNN probability logs
# - `SUMMARY.txt` - Training summary

# %%
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print("\nResults saved to: kaggle_results.zip")
print("\nTo download:")
print("1. Click on 'Output' tab in Kaggle")
print("2. Download 'kaggle_results.zip'")
print("\nTo continue training locally:")
print("1. Extract kaggle_results.zip")
print("2. Copy weights/ and runs/ to your local project")
print("3. Run: uv run python train.py --load best")
print("="*60)
