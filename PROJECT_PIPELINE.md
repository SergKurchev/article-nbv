# NBV Project Pipeline - Complete Guide

**Date**: 2026-04-22  
**Version**: 1.0

---

## Project Overview

Next-Best-View (NBV) system using Deep RL for active object classification with 3 progressive training stages.

---

## Stage 1: Single Object (Baseline)

### Idea
Train baseline NBV agent on simplest scenario - one object at fixed position. Establish performance benchmarks.

### Configuration
```python
# config.py
SCENE_STAGE = 1
NUM_CLASSES = 8  # 8 primitive shapes
OBJECT_MODE = "primitives"
```

### Implementation Details
- **Scene**: 1 object at position `[0.5, 0.0, z]` where `z = SCENE_BOUNDS_Z_MIN + half_height + 0.05`
- **Obstacles**: 0
- **Object placement**: Fixed XY, Z calculated from object height
- **Textures**: 3 classes (red, mixed gradient, green)

### Full Pipeline

#### 1. Generate Dataset
```bash
# Generate RGB-D dataset (8000 samples × 20 views)
uv run python src/vision/dataset.py

# Output: dataset/primitives/sample_XXXXX/
#   - rgb/00000.png - 00019.png
#   - depth/00000.npy - 00019.npy
#   - masks/00000.png - 00019.png
#   - cameras.json
#   - color_map.json
```

**Parameters** (config.py):
```python
DATASET_SAMPLES_PER_CLASS = 1000
DATASET_VIEWS_PER_SAMPLE = 20
DATASET_CAMERA_RADIUS_MIN = 0.3
DATASET_CAMERA_RADIUS_MAX = 0.6
```

#### 2. Train Vision Model
```bash
# Train MultiModalNet (RGB-D + robot state)
uv run python src/vision/train_cnn.py

# Output: weights/primitives/multimodal_best.pt
```

**Parameters**:
```python
CNN_ARCHITECTURE = "MultiModalNet"  # or "LightweightODIN"
CNN_BATCH_SIZE = 16
CNN_EPOCHS = 10
CNN_LR = 1e-3
```

#### 3. Train RL Agent
```bash
# Headless training (fast)
uv run python train.py

# With GUI (slow, for debugging)
uv run python train.py --gui

# Output: runs/rl_train_v1.0_primitives_8obj_YYYYMMDD_HHMMSS/
#   - best_policy.zip
#   - last_policy.zip
#   - logs.jsonl
```

**Parameters**:
```python
TOTAL_TIMESTEPS = 500000
BATCH_SIZE = 128
LEARNING_RATE = 1e-2
MAX_STEPS_PER_EPISODE = 10
REWARD_SCALE = 10.0
PENALTY_COLLISION = -15.0
```

#### 4. Evaluate Agent
```bash
# Evaluate trained policy
uv run python evaluate.py

# Interactive GUI
uv run python gui.py
```

### Debug Commands

```bash
# Test Stage 1 scene generation
uv run python scripts/test_scene_stages.py --stage 1

# Validate with screenshots
uv run python scripts/validate_stages.py --stage 1

# Check dataset integrity
ls dataset/primitives/ | wc -l  # Should be 8000

# Test CNN loading
uv run python -c "from src.vision.models import NetLoader; model = NetLoader('MultiModalNet', 8); print('OK')"

# Test environment
uv run python -c "from src.simulation.environment import NBVEnv; env = NBVEnv(); print('OK')"
```

---

## Stage 2: Multiple Objects

### Idea
Train agent to handle multiple objects (2-10) with uniform spatial distribution. Learn to classify multiple targets and aggregate uncertainty.

### Configuration
```python
# config.py
SCENE_STAGE = 2
MIN_OBJECTS = 2
MAX_OBJECTS = 10
```

### Implementation Details
- **Scene**: 2-10 objects randomly placed
- **Obstacles**: 0
- **Object placement**: Uniform distribution in bounds with collision detection
- **Collision detection**: Uses real bounding boxes from mesh.json
- **Min distance**: `object_radius_1 + object_radius_2 + 0.05m`

### Spatial Bounds
```python
SCENE_BOUNDS_X_MIN = 0.2
SCENE_BOUNDS_X_MAX = 0.8
SCENE_BOUNDS_Y_MIN = -0.3
SCENE_BOUNDS_Y_MAX = 0.3
SCENE_BOUNDS_Z_MIN = 0.15
SCENE_BOUNDS_Z_MAX = 0.4
```

### Full Pipeline

#### 1. Generate Multi-Object Dataset
```bash
# Set SCENE_STAGE = 2 in config.py
uv run python src/vision/dataset.py

# Output: Same structure as Stage 1, but with multiple objects per sample
```

#### 2. Train Vision Model
```bash
# Train on multi-object scenes
uv run python src/vision/train_cnn.py

# Model learns to classify multiple objects simultaneously
```

#### 3. Train RL Agent
```bash
# Train with multi-object reward
uv run python train.py

# Reward aggregates accuracy across all objects
```

**Reward Function**:
```python
# Average accuracy improvement across all objects
reward = mean([acc_diff_obj1, acc_diff_obj2, ...]) * REWARD_SCALE
reward += PENALTY_COLLISION if collision else 0
reward += PENALTY_OOB if out_of_bounds else 0
```

#### 4. Evaluate
```bash
uv run python evaluate.py

# Metrics: per-object accuracy, total scene accuracy, steps to convergence
```

### Debug Commands

```bash
# Test Stage 2 scene generation
uv run python scripts/test_scene_stages.py --stage 2

# Validate collision-free placement
uv run python scripts/validate_stages.py --stage 2

# Check object count distribution
uv run python -c "
from src.simulation.asset_loader import AssetLoader
import pybullet as p
import config
config.SCENE_STAGE = 2
client = p.connect(p.DIRECT)
loader = AssetLoader(client)
for i in range(100):
    objs = loader.generate_scene()
    print(f'Run {i}: {len(objs)} objects')
    loader.clear_scene()
"

# Visualize multi-object scene
uv run python gui.py
```

---

## Stage 3: Multiple Objects + Obstacles

### Idea
Train agent to handle occlusions from obstacles. Learn active search and view planning around obstacles.

### Configuration
```python
# config.py
SCENE_STAGE = 3
MIN_OBJECTS = 2
MAX_OBJECTS = 10
MIN_OBSTACLES = 1
MAX_OBSTACLES = 5
```

### Implementation Details
- **Scene**: 2-10 objects + 1-5 obstacles
- **Obstacles**: Gray thin panels (random dimensions)
- **Placement**: Uniform distribution with collision detection
- **Collision checks**: 
  - Object ↔ Object
  - Object ↔ Obstacle
  - Obstacle ↔ Obstacle
  - All ↔ Robot base (0.25m safety)

### Obstacle Dimensions
```python
# Random per obstacle
half_extents = [
    random.uniform(0.01, 0.05),  # Thickness
    random.uniform(0.1, 0.2),    # Width
    random.uniform(0.1, 0.3)     # Height
]
```

### Full Pipeline

#### 1. Generate Occluded Dataset
```bash
# Set SCENE_STAGE = 3 in config.py
uv run python src/vision/dataset.py

# Dataset includes partial occlusions from obstacles
```

#### 2. Train Vision Model
```bash
# Train on occluded scenes
uv run python src/vision/train_cnn.py

# Model learns to handle partial visibility
```

#### 3. Train RL Agent
```bash
# Train with obstacle avoidance
uv run python train.py

# Agent learns to navigate around obstacles
```

**Reward Function**:
```python
# Same as Stage 2, but with stricter collision penalty
reward = mean([acc_diff_obj1, acc_diff_obj2, ...]) * REWARD_SCALE
reward += PENALTY_COLLISION if collision else 0  # -15.0
reward += PENALTY_OOB if out_of_bounds else 0    # -10.0
```

#### 4. Evaluate
```bash
uv run python evaluate.py

# Metrics: occlusion handling, obstacle avoidance rate, view diversity
```

### Debug Commands

```bash
# Test Stage 3 scene generation
uv run python scripts/test_scene_stages.py --stage 3

# Validate collision-free placement
uv run python scripts/validate_stages.py --stage 3

# Check obstacle placement success rate
uv run python -c "
from src.simulation.asset_loader import AssetLoader
import pybullet as p
import config
config.SCENE_STAGE = 3
client = p.connect(p.DIRECT)
loader = AssetLoader(client)
for i in range(100):
    objs = loader.generate_scene()
    print(f'Run {i}: {len(objs)} objects, {len(loader.obstacles)} obstacles')
    loader.clear_scene()
"

# Visualize occlusion scenarios
uv run python gui.py
```

---

## Cross-Stage Commands

### Test All Stages
```bash
# Test scene generation for all stages
uv run python scripts/test_scene_stages.py --all

# Validate all stages with screenshots
uv run python scripts/validate_stages.py --all
```

### Compare Performance
```bash
# Train on each stage and compare
for stage in 1 2 3; do
    # Update config.py: SCENE_STAGE = $stage
    uv run python train.py
    uv run python evaluate.py
done

# Compare metrics from runs/ directory
```

### Ablation Studies
```bash
# Test different CNN architectures
for arch in "MultiModalNet" "LightweightODIN" "SimpleNet"; do
    # Update config.py: CNN_ARCHITECTURE = $arch
    uv run python src/vision/train_cnn.py
    uv run python train.py
done

# Test different reward scales
for scale in 5.0 10.0 20.0; do
    # Update config.py: REWARD_SCALE = $scale
    uv run python train.py
done
```

---

## PointNet Pipeline (Alternative Vision Model)

### Idea
Use 3D point clouds instead of RGB-D images for classification.

### Full Pipeline

#### 1. Generate RGB-D Dataset
```bash
# Same as Stage 1-3
uv run python src/vision/dataset.py
```

#### 2. Convert to Point Clouds
```bash
# Convert RGB-D to point clouds (1024 points per sample)
uv run python src/vision/dataset_pointnet.py

# Output: dataset/primitives_pointnet/
#   - pointclouds/sample_XXXXX.npy (1024, 3)
#   - labels.json
```

**Parameters**:
```python
POINTNET_NUM_POINTS = 1024
```

#### 3. Train PointNet
```bash
# Train PointNet classifier
uv run python src/vision/train_pointnet.py

# Output: weights/primitives/pointnet_best.pt
```

**Parameters**:
```python
POINTNET_BATCH_SIZE = 32
POINTNET_LR = 1e-3
POINTNET_EPOCHS = 50
```

#### 4. Test PointNet
```bash
# Test trained model
uv run python scripts/test_pointnet.py

# With visualization
uv run python scripts/test_pointnet.py --visualize
```

### Debug Commands
```bash
# Check point cloud shape
uv run python -c "
import numpy as np
pc = np.load('dataset/primitives_pointnet/pointclouds/sample_00000.npy')
print(f'Shape: {pc.shape}')  # Should be (1024, 3)
print(f'Range: [{pc.min():.3f}, {pc.max():.3f}]')
"

# Verify labels
uv run python -c "
import json
with open('dataset/primitives_pointnet/labels.json') as f:
    labels = json.load(f)
print(f'Total samples: {len(labels)}')
print(f'Classes: {set(labels.values())}')  # Should be {0, 1, ..., 7}
"
```

---

## Common Debug Commands

### Check PyBullet
```bash
uv run python -c "import pybullet as p; print('PyBullet OK')"
```

### Check GPU
```bash
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Check Dataset Size
```bash
# Count samples
ls dataset/primitives/ | wc -l

# Check disk usage
du -sh dataset/
```

### Check Model Weights
```bash
# List saved models
ls -lh weights/primitives/

# Load and inspect
uv run python -c "
import torch
model = torch.load('weights/primitives/multimodal_best.pt')
print(f'Keys: {model.keys()}')
"
```

### Monitor Training
```bash
# Watch logs in real-time
tail -f runs/rl_train_*/logs.jsonl

# Parse metrics
uv run python -c "
import json
with open('runs/rl_train_*/logs.jsonl') as f:
    for line in f:
        data = json.loads(line)
        print(f\"Step {data['step']}: reward={data['reward']:.2f}\")
"
```

### Profile Performance
```bash
# Time dataset generation
time uv run python src/vision/dataset.py

# Time CNN training
time uv run python src/vision/train_cnn.py

# Time RL training (1000 steps)
# Update config.py: TOTAL_TIMESTEPS = 1000
time uv run python train.py
```

---

## Troubleshooting

### Issue: Objects touching ground
**Solution**: Check `SCENE_BOUNDS_Z_MIN` and safety margin in `asset_loader.py`
```python
z_position = SCENE_BOUNDS_Z_MIN + half_height + 0.05  # 5cm safety
```

### Issue: Too many placement failures
**Solution**: Increase bounds or reduce object count
```python
SCENE_BOUNDS_X_MAX = 1.0  # Increase from 0.8
MAX_OBJECTS = 8  # Reduce from 10
```

### Issue: CNN not converging
**Solution**: Check learning rate and batch size
```python
CNN_LR = 1e-4  # Reduce from 1e-3
CNN_BATCH_SIZE = 32  # Increase from 16
```

### Issue: RL agent colliding
**Solution**: Increase collision penalty
```python
PENALTY_COLLISION = -20.0  # Increase from -15.0
```

### Issue: Out of memory
**Solution**: Reduce batch sizes
```python
CNN_BATCH_SIZE = 8
BATCH_SIZE = 64
```

---

## Performance Benchmarks

### Expected Training Times (CPU)
- Dataset generation: ~15 min (8000 samples)
- CNN training: ~10 min (10 epochs)
- RL training: ~2-3 hours (500k steps)
- PointNet training: ~30 min (50 epochs)

### Expected Metrics
- **Stage 1**: 
  - CNN accuracy: >90%
  - RL reward: >5.0
  - Steps to convergence: <5
  
- **Stage 2**:
  - CNN accuracy: >85%
  - RL reward: >3.0
  - Steps to convergence: <7
  
- **Stage 3**:
  - CNN accuracy: >80%
  - RL reward: >2.0
  - Steps to convergence: <10

---

## File Structure Reference

```
NBV_with_obstacles_and_robot/
├── config.py                    # ALL parameters here
├── train.py                     # RL training
├── evaluate.py                  # Evaluation
├── gui.py                       # Interactive GUI
│
├── src/
│   ├── simulation/
│   │   ├── environment.py       # NBV Gym environment
│   │   ├── asset_loader.py      # Scene generation (3 stages)
│   │   ├── camera.py            # Camera with intrinsics
│   │   └── robot.py             # Kuka manipulator
│   │
│   ├── vision/
│   │   ├── models.py            # MultiModalNet, LightweightODIN, PointNet
│   │   ├── dataset.py           # RGB-D dataset generation
│   │   ├── dataset_pointnet.py  # Point cloud conversion
│   │   ├── train_cnn.py         # CNN training
│   │   └── train_pointnet.py    # PointNet training
│   │
│   └── rl/
│       ├── agent.py             # SAC agent wrapper
│       └── callbacks.py         # Training callbacks
│
├── scripts/
│   ├── test_scene_stages.py    # Test scene generation
│   ├── validate_stages.py      # Screenshot validation
│   └── test_pointnet.py        # PointNet testing
│
├── dataset/primitives/          # RGB-D dataset
├── weights/primitives/          # Trained models
└── runs/                        # Training logs
```

---

**Last Updated**: 2026-04-22 15:30 UTC  
**Status**: Production Ready
