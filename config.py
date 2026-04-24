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

# --- Scene Generation Stage ---
# Stage 1: Single object, no obstacles
# Stage 2: Multiple objects (2-10), no obstacles
# Stage 3: Multiple objects (2-10) + obstacles
SCENE_STAGE = 1  # Options: 1, 2, 3

# Stage 2 & 3: Multi-object parameters
MIN_OBJECTS = 2  # Minimum number of objects in multi-object scenes
MAX_OBJECTS = 10  # Maximum number of objects in multi-object scenes

# Stage 3: Obstacle parameters
MIN_OBSTACLES = 1  # Minimum number of obstacles
MAX_OBSTACLES = 5  # Maximum number of obstacles

# Spatial distribution bounds for Stage 2 & 3
# Objects and obstacles are placed within this volume
SCENE_BOUNDS_X_MIN = 0.2  # Minimum X coordinate
SCENE_BOUNDS_X_MAX = 0.8  # Maximum X coordinate
SCENE_BOUNDS_Y_MIN = -0.3  # Minimum Y coordinate
SCENE_BOUNDS_Y_MAX = 0.3  # Maximum Y coordinate
SCENE_BOUNDS_Z_MIN = 0.15  # Minimum Z coordinate (above ground) - raised to prevent ground collision
SCENE_BOUNDS_Z_MAX = 0.4  # Maximum Z coordinate

# Collision detection for object placement
SCENE_MIN_OBJECT_DISTANCE = 0.25  # Minimum distance between objects (meters) - increased for safety
SCENE_MAX_PLACEMENT_ATTEMPTS = 100  # Maximum attempts to place object without collision - increased

# Object scaling
OBJECT_SCALE_FACTOR = 1.0  # Scale factor for object size (1.0 = default size)

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

# Mixed texture pre-generation
TEXTURE_NUM_MIXED_VARIANTS = 20  # Number of different mixed textures to pre-generate
TEXTURE_MIXED_RANDOM_SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021,
                               2223, 2425, 2627, 2829, 3031, 3233, 3435, 3637, 3839, 4041]  # Seeds for reproducible mixed textures

# Gradient curve parameters
TEXTURE_GRADIENT_TYPE = "curved"  # "linear" or "curved"
TEXTURE_GRADIENT_ANGLE = 45.0  # Base angle for linear gradient (degrees)
TEXTURE_GRADIENT_SHARPNESS = 0.6  # 0.0=smooth, 1.0=sharp transition
TEXTURE_GRADIENT_STEEPNESS = 30  # Sigmoid steepness for 3D rendering (20-50, higher=sharper)

# Curved gradient parameters
TEXTURE_CURVE_COMPLEXITY = 10  # Number of control points for Bezier curve (2-10, higher=more curves)
TEXTURE_CURVE_AMPLITUDE = 0.7  # Maximum curve deviation (0.0-0.5)
TEXTURE_MIN_AREA_RATIO = 0.3  # Minimum area for each color (30%)
TEXTURE_MAX_AREA_RATIO = 0.7  # Maximum area for each color (70%)

# Curve positioning (to ensure gradient reaches edges)
TEXTURE_CURVE_START_X_MIN = 0.1  # Minimum horizontal position for curve start (0.0-1.0)
TEXTURE_CURVE_START_X_MAX = 0.9  # Maximum horizontal position for curve start (0.0-1.0)
TEXTURE_CURVE_END_X_MIN = 0.1  # Minimum horizontal position for curve end (0.0-1.0)
TEXTURE_CURVE_END_X_MAX = 0.9  # Maximum horizontal position for curve end (0.0-1.0)
TEXTURE_CURVE_EXTENSION_TOP = 0.2  # Extend curve beyond top edge (fraction of size)
TEXTURE_CURVE_EXTENSION_BOTTOM = 0.2  # Extend curve beyond bottom edge (fraction of size)

# Bezier curve evaluation
TEXTURE_BEZIER_SAMPLES = 200  # Number of points to sample along Bezier curve (higher=smoother)

# Distance field optimization
TEXTURE_DOWNSAMPLE_FACTOR = 4  # Downsample factor for distance field computation (2=4x faster, 3=9x faster, 4=16x faster)

# Visualization
TEXTURE_DEBUG_CURVE_WIDTH = 2  # Width of curve line in debug visualization
TEXTURE_DEBUG_CURVE_SKIP = 5  # Draw every Nth point of curve (lower=more detailed)
TEXTURE_DEBUG_CONTROL_POINT_RADIUS = 5  # Radius of control point circles in debug

# Generation
TEXTURE_MAX_GENERATION_ATTEMPTS = 50  # Maximum attempts to generate valid texture

# Realism
TEXTURE_NOISE_LEVEL = 0.0  # No noise for clean textures (0.0-0.2)

# --- CNN Training ---
CNN_ARCHITECTURE = "MultiModalNet"  # Options: "MultiModalNet", "LightweightODIN", "SimpleNet", "PointNet"
CNN_LR = 1e-3
CNN_BATCH_SIZE = 16
CNN_EPOCHS = 10
CNN_VAL_SPLIT = 0.2

# --- PointNet Training ---
POINTNET_NUM_POINTS = 1024  # Number of points to sample from each point cloud
POINTNET_BATCH_SIZE = 32  # Batch size for PointNet training
POINTNET_LR = 1e-3  # Learning rate
POINTNET_EPOCHS = 50  # Number of training epochs

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
