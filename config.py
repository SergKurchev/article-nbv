import os
import ctypes
from pathlib import Path

def get_short_path(path_str):
    if os.name == 'nt' and hasattr(ctypes, 'windll'):
        try:
            GetShortPathName = ctypes.windll.kernel32.GetShortPathNameW
            buf = ctypes.create_unicode_buffer(1024)
            res = GetShortPathName(str(path_str), buf, 1024)
            if res > 0:
                return buf.value
        except Exception:
            pass
    return str(path_str)

# --- Paths ---
BASE_DIR = Path(get_short_path(Path(__file__).resolve().parent))
SRC_DIR = BASE_DIR / "src"
DATA_DIR = SRC_DIR / "data"
RUNS_DIR = BASE_DIR / "runs"

# --- Project Version ---
VERSION = "v1.0"

def get_run_dir(task_type):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{task_type}_{VERSION}_{OBJECT_MODE}_{NUM_CLASSES}obj_{timestamp}"
    path = RUNS_DIR / folder_name
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_latest_rl_run_dir():
    prefix = f"rl_train_{VERSION}_{OBJECT_MODE}_{NUM_CLASSES}obj_"
    if not RUNS_DIR.exists():
        return None
    runs = [d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    if not runs:
        return None
    runs.sort(key=lambda x: x.name)
    return runs[-1]

ROBOT_URDF = "kuka_iiwa/model.urdf"

# --- Object Dataset Mode ---
OBJECT_MODE = "primitives" # Choose "complex" or "primitives"

# --- Simulation & Env ---
# Number of categories
if OBJECT_MODE == "complex":
    NUM_CLASSES = 5
else:
    NUM_CLASSES = 8

OBJECTS_DIR = DATA_DIR / "objects" / OBJECT_MODE
DATASET_DIR = BASE_DIR / "dataset" / OBJECT_MODE
TEXTURE_PATH = DATA_DIR / "objects" / "texture.png" # Common texture at root

# Obstacles per reset
MIN_OBSTACLES = 0
MAX_OBSTACLES = 0
# Episode limits
MAX_STEPS_PER_EPISODE = 10

# Camera
IMAGE_SIZE = 224
IMAGE_CHANNELS = 4 # RGB + Depth
CAMERA_POS = [1.0, 0.0, 0.5]
CAMERA_TARGET = [0.0, 0.0, 0.2]
CAMERA_UP = [0.0, 0.0, 1.0]

# --- RL Training (SAC) ---
TOTAL_TIMESTEPS = 500000
BATCH_SIZE = 128
LEARNING_RATE = 1e-2
BUFFER_SIZE = 10000
LEARNING_STARTS = 1000

# Absolute action space bounds (X, Y, Z, roll, pitch, yaw)
ACTION_MIN = [0.2, -0.5, 0.0, -3.14, -3.14, -3.14]
ACTION_MAX = [0.8, 0.5, 0.8, 3.14, 3.14, 3.14]

# Reward shaping
REWARD_SCALE = 10.0
PENALTY_OOB = -10.0
PENALTY_COLLISION = -15.0  # Отдельный (более строгий) штраф за столкновение

# Evaluation & Callbacks
EVAL_FREQ = 2000
N_EVAL_EPISODES = 10
MIN_EPISODES_FOR_VIDEO = 100
VIDEO_EPISODES = 3
PLOT_MOVING_AVERAGE_WINDOW = 50

# --- Dataset Generation ---
DATASET_SAMPLES_PER_CLASS = 1000
DATASET_VIEWS_PER_SAMPLE = 5  # Number of camera views around each object
DATASET_MIN_VALID_VIEWS = 5  # Minimum valid views to keep a sample
DATASET_MIN_OBJECT_PIXELS = 50  # Minimum visible pixels to consider view valid
DATASET_PLACEMENT_ATTEMPTS = 20  # Max attempts to find collision-free placement
DATASET_COLLISION_MARGIN = 0.005  # Collision detection margin in meters

# Camera orbit parameters for dataset generation
DATASET_CAMERA_RADIUS_MIN = 0.3  # Minimum camera distance from target
DATASET_CAMERA_RADIUS_MAX = 0.6  # Maximum camera distance from target
DATASET_CAMERA_PHI_MIN = 0.1  # Minimum elevation angle (radians)
DATASET_CAMERA_PHI_MAX = 1.37  # Maximum elevation angle (pi/2 - 0.2 radians)
DATASET_CAMERA_THETA_JITTER = 0.1  # Random jitter for azimuth angle (radians)

# Target object position for dataset generation
DATASET_TARGET_POS = [0.5, 0.0, 0.2]  # Match environment position

# Color mapping for instance segmentation
DATASET_COLOR_TARGET = [255, 0, 0]  # Red for target object
DATASET_COLOR_ROBOT = [0, 255, 0]  # Green for robot
DATASET_COLOR_OBSTACLE_BASE = [0, 0, 255]  # Blue base for obstacles
DATASET_COLOR_OBSTACLE_STEP = 10  # Color step between obstacles

# --- Texture Generation ---
# Texture parameters
TEXTURE_SIZE = 512
TEXTURE_RED_COLOR = [255, 0, 0]
TEXTURE_GREEN_COLOR = [0, 255, 0]

# Gradient curve parameters
TEXTURE_GRADIENT_TYPE = "curved"  # "linear" or "curved"
TEXTURE_GRADIENT_ANGLE = 45.0  # Base angle for linear gradient (degrees)
TEXTURE_GRADIENT_SHARPNESS = 0.5  # 0.0=smooth, 1.0=sharp transition

# Curved gradient parameters
TEXTURE_CURVE_COMPLEXITY = 3  # Number of control points for Bezier curve (2-5)
TEXTURE_CURVE_AMPLITUDE = 0.3  # Maximum curve deviation (0.0-0.5)
TEXTURE_MIN_AREA_RATIO = 0.3  # Minimum area for each color (30%)
TEXTURE_MAX_AREA_RATIO = 0.7  # Maximum area for each color (70%)

# Realism
TEXTURE_NOISE_LEVEL = 0.05  # Add realistic noise (0.0-0.2)

# --- CNN Training ---
CNN_ARCHITECTURE = "MultiModalNet"  # Options: "MultiModalNet", "LightweightODIN", "SimpleNet"
CNN_LR = 1e-3
CNN_BATCH_SIZE = 16
CNN_EPOCHS = 10
CNN_VAL_SPLIT = 0.2

CNN_WEIGHTS_RUN_NAME = None # Set to e.g. 'cnn_train_v1.0_5obj_20260323_202140' to load weights from a specific run, or None to use default 'weights' directory
if CNN_WEIGHTS_RUN_NAME is not None:
    CNN_WEIGHTS_DIR = RUNS_DIR / CNN_WEIGHTS_RUN_NAME
else:
    CNN_WEIGHTS_DIR = BASE_DIR / "weights" / OBJECT_MODE

CNN_BEST_MODEL_NAME = "multimodal_best.pt"
CNN_LAST_MODEL_NAME = "multimodal_last.pt"
CNN_MODEL_PATH = CNN_WEIGHTS_DIR / CNN_BEST_MODEL_NAME
CNN_LOAD_MODE = "best" # "best" or "none"
CNN_RESUME = False

# Flags parsing
import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "eval", "dataset"], default="train")
    parser.add_argument("--gui", action="store_true", help="Run with GUI visualizer (slow). Default is headless.")
    parser.add_argument("--no_arm", action="store_true", help="Run without robot arm (object flies).")
    parser.add_argument("--load", type=str, choices=["none", "best", "last"], default="none")
    return parser.parse_args()
