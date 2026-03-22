import os
import ctypes
from pathlib import Path

def get_short_path(path_str):
    if os.name == 'nt':
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

ROBOT_URDF = "kuka_iiwa/model.urdf"
OBJECTS_DIR = DATA_DIR / "objects"
TEXTURE_PATH = OBJECTS_DIR / "texture.png"

# --- Simulation & Env ---
# Number of categories
NUM_CLASSES = 18
# Obstacles per reset
MIN_OBSTACLES = 3
MAX_OBSTACLES = 7
# Episode limits
MAX_STEPS_PER_EPISODE = 10

# Camera
IMAGE_SIZE = 224
IMAGE_CHANNELS = 4 # RGB + Depth
CAMERA_POS = [1.0, 0.0, 0.5]
CAMERA_TARGET = [0.0, 0.0, 0.2]
CAMERA_UP = [0.0, 0.0, 1.0]

# --- RL Training (SAC) ---
TOTAL_TIMESTEPS = 50000
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
BUFFER_SIZE = 100000
LEARNING_STARTS = 1000

# Absolute action space bounds (X, Y, Z, roll, pitch, yaw)
ACTION_MIN = [0.2, -0.5, 0.0, -3.14, -3.14, -3.14]
ACTION_MAX = [0.8, 0.5, 0.8, 3.14, 3.14, 3.14]

# Reward shaping
REWARD_SCALE = 10.0
PENALTY_OOB = -10.0

# Evaluation & Callbacks
EVAL_FREQ = 2000
N_EVAL_EPISODES = 10
MIN_EPISODES_FOR_VIDEO = 100
VIDEO_EPISODES = 3

# --- CNN Training (Vision) ---
CNN_LR = 1e-3
CNN_BATCH_SIZE = 16
CNN_EPOCHS = 10
CNN_VAL_SPLIT = 0.2
CNN_WEIGHTS_DIR = BASE_DIR / "weights"
CNN_MODEL_NAME = "multimodal_best.pt"
CNN_MODEL_PATH = CNN_WEIGHTS_DIR / CNN_MODEL_NAME
CNN_LOAD_MODE = "best" # "best" or "none"
CNN_RESUME = False

# Flags parsing
import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "eval", "dataset"], default="train")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (DIRECT).")
    parser.add_argument("--no_arm", action="store_true", help="Run without robot arm (object flies).")
    parser.add_argument("--load", type=str, choices=["none", "best", "last"], default="none")
    return parser.parse_args()
