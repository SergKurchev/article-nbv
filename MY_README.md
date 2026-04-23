# NBV Project - Complete Guide

## Quick Start

### Generate Datasets (8 samples for testing)
```bash
uv run python scripts/prepare_stage_datasets.py --stage 1 --samples 8
uv run python scripts/prepare_stage_datasets.py --stage 2 --samples 8
uv run python scripts/prepare_stage_datasets.py --stage 3 --samples 8
```

### Generate Full Datasets (1000 samples per class)
```bash
uv run python scripts/prepare_stage_datasets.py --stage 1 --samples 8000
uv run python scripts/prepare_stage_datasets.py --stage 2 --samples 8000
uv run python scripts/prepare_stage_datasets.py --stage 3 --samples 8000
```

### Generate 3D Visualizations
```bash
# First sample of each stage
uv run python scripts/generate_all_visualizations.py --max-samples 1

# Open in browser
start dataset/primitives/stage1/sample_00000/visualization.html
```

### Train Models
```bash
# CNN training
uv run python src/vision/train_cnn.py

# RL training
uv run python train.py

# Evaluation
uv run python evaluate.py
```

### Update Kaggle Datasets
```bash
# Check what will be uploaded
uv run python scripts/update_kaggle_datasets.py --dry-run

# Upload all stages
uv run python scripts/update_kaggle_datasets.py
```

---

## Training Stages

### Stage 1: Single Object (Baseline)
- **Objects**: 1 at fixed position `[0.5, 0.0, z]`
- **Obstacles**: 0
- **Purpose**: Baseline performance

### Stage 2: Multiple Objects
- **Objects**: 2-10 with uniform spatial distribution
- **Obstacles**: 0
- **Purpose**: Multi-object classification

### Stage 3: Multiple Objects + Obstacles
- **Objects**: 2-10 with uniform spatial distribution
- **Obstacles**: 1-5 (gray panels)
- **Purpose**: Occlusion handling

---

## Dataset Structure

Each sample contains:
```
sample_XXXXX/
├── rgb/                    # RGB images (224×224 PNG)
│   ├── 00000.png
│   └── ...
├── depth/                  # Depth maps (224×224 NPY, float32, meters)
│   ├── 00000.npy
│   └── ...
├── masks/                  # Instance segmentation (224×224 PNG, colored)
│   ├── 00000.png
│   └── ...
├── labels/                 # YOLO bounding boxes (TXT)
│   ├── 00000.txt
│   └── ...
├── cameras.json           # Camera metadata (position, target, rotation, intrinsics)
└── color_map.json         # Instance ID → Category mapping
```

---

## 3D Visualization

### Controls
- **Scroll**: Zoom in/out
- **Left drag**: Orbit around objects
- **Right drag / Middle**: Pan camera

### Display Modes
- **RGB**: Color as seen by camera
- **Category**: Color by category (target/robot/obstacles)
- **Instances**: Color by instance (each object different color)

### Additional Options
- **Cameras**: Show/hide camera frustums
- **Cam Dots**: Show/hide camera position dots
- **Reset View**: Return to initial camera position

---

## Kaggle Datasets

- **Stage 1**: https://www.kaggle.com/datasets/sergeykurchev/nbv-stage1-dataset
- **Stage 2**: https://www.kaggle.com/datasets/sergeykurchev/nbv-stage2-dataset
- **Stage 3**: https://www.kaggle.com/datasets/sergeykurchev/nbv-stage3-dataset

### Download from Kaggle
```bash
pip install kaggle

kaggle datasets download -d sergeykurchev/nbv-stage1-dataset
kaggle datasets download -d sergeykurchev/nbv-stage2-dataset
kaggle datasets download -d sergeykurchev/nbv-stage3-dataset

unzip nbv-stage1-dataset.zip -d dataset/primitives/stage1/
unzip nbv-stage2-dataset.zip -d dataset/primitives/stage2/
unzip nbv-stage3-dataset.zip -d dataset/primitives/stage3/
```

---

## Technical Details

### Camera Coordinates (OpenGL Convention)
- **Forward**: -Z
- **Up**: +Y
- **Right**: +X

### Depth Unprojection
```python
X_cam = (u - cx) * depth / fx
Y_cam = -(v - cy) * depth / fy  # Flip Y for OpenGL
Z_cam = -depth  # Forward is -Z
```

### Primitive Normals
All primitives use **counter-clockwise (CCW) winding order** when viewed from outside, ensuring normals point outward.

### Color Scheme (Category Mode)
- 🔴 **Red** - Target objects
- 🟢 **Green** - Robot (Kuka manipulator)
- 🔵 **Blue** - Obstacles (Stage 3 only)

---

## Key Scripts

### Dataset Generation
- `scripts/prepare_stage_datasets.py` - Generate and verify datasets
- `scripts/master_pipeline.py` - Full pipeline for all stages
- `src/data/generate_primitives.py` - Generate 8 primitive shapes

### Visualization
- `scripts/generate_sample_viewer.py` - Generate 3D visualization for one sample
- `scripts/generate_all_visualizations.py` - Batch generate visualizations

### Kaggle Upload
- `scripts/update_kaggle_datasets.py` - Automatic version updater

### Training
- `src/vision/train_cnn.py` - CNN training
- `train.py` - RL training
- `evaluate.py` - Evaluation
- `gui.py` - Interactive GUI

---

## Troubleshooting

### Visualization Issues
- **Objects not visible**: Press "Reset View" or try "Category" mode
- **Slow loading**: Reduce points with `--stride 4 --max-points 200000`
- **Browser compatibility**: Use Chrome 90+, Firefox 88+, or Edge 90+

### Dataset Generation
- **Objects touching ground**: Check `SCENE_BOUNDS_Z_MIN` and safety margin
- **Too many placement failures**: Increase bounds or reduce object count
- **CNN not converging**: Reduce learning rate or increase batch size

### Kaggle Upload
- **401 Unauthorized**: Check credentials in `~/.kaggle/kaggle.json`
- **Dataset not found**: Script will auto-create on first upload

---

## Changelog

### 2026-04-22 - Critical Fixes
- ✅ Fixed Stage 2: Now generates 2-10 objects WITHOUT obstacles
- ✅ Fixed Stage 3: Now generates 2-10 objects WITH obstacles (1-5)
- ✅ Fixed visualization: Objects no longer duplicate (uses correct camera target/up)
- ✅ Created stage-aware dataset generator (`src/vision/dataset_stage.py`)
- ✅ Added class tracking to `AssetLoader.generate_scene()`
- ✅ Removed all old incorrect datasets

---

**Version**: 1.0  
**Date**: 2026-04-23  
**Status**: Production Ready ✅
