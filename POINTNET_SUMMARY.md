# PointNet Implementation Summary

**Date**: 2026-04-22
**Status**: Completed

## Overview

Successfully implemented PointNet architecture for 3D point cloud classification in the NBV project.

---

## Completed Tasks

### 1. PointNet Architecture ✅

**File**: `src/vision/models.py`

**Components implemented:**
- `TNet` class - Transformation network for rotation invariance
  - Input transform (3x3 matrix)
  - Feature transform (64x64 matrix)
- `PointNet` class - Main classification network
  - Hierarchical MLP layers (3→64→64→64→128→1024)
  - Symmetric max pooling for permutation invariance
  - Vector fusion for robot state integration
  - Classification head with dropout

**Parameters**: ~3.5M trainable parameters

### 2. Dataset Generator ✅

**File**: `src/vision/dataset_pointnet.py`

**Features:**
- Converts RGB-D images to 3D point clouds
- Back-projection using camera intrinsics
- Point cloud normalization (center + unit sphere)
- Fixed-size sampling (1024 points)
- Saves in efficient numpy format

**Output structure:**
```
dataset_pointnet/primitives/
├── sample_00000/
│   ├── point_clouds.npy    # [num_views, 1024, 3]
│   ├── vectors.npy         # [num_views, 15]
│   └── metadata.json       # class_id, num_views
└── ...
```

**Statistics:**
- Processed: 8000 samples
- Skipped: 0 samples
- Processing time: ~15 minutes

### 3. Training Script ✅

**File**: `src/vision/train_pointnet.py`

**Features:**
- PyTorch DataLoader with train/val split (80/20)
- Adam optimizer with StepLR scheduler
- CrossEntropyLoss
- Best model checkpointing
- Progress tracking with tqdm

**Configuration:**
- Batch size: 32
- Learning rate: 1e-3
- Epochs: 50
- Scheduler: step_size=20, gamma=0.5

### 4. Test Script ✅

**File**: `scripts/test_pointnet.py`

**Features:**
- Load trained model
- Test on random samples
- Multi-view prediction aggregation
- 3D point cloud visualization (matplotlib)
- Accuracy and confidence metrics

### 5. Configuration ✅

**File**: `config.py`

**Added parameters:**
```python
CNN_ARCHITECTURE = "PointNet"  # Added to options
POINTNET_NUM_POINTS = 1024
POINTNET_BATCH_SIZE = 32
POINTNET_LR = 1e-3
POINTNET_EPOCHS = 50
```

### 6. Documentation ✅

**Files created:**
- `POINTNET_GUIDE.md` - Comprehensive usage guide
- `POINTNET_SUMMARY.md` - This file

---

## Technical Details

### Point Cloud Conversion

**Depth to 3D:**
```python
x = (u - cx) * z / fx
y = (v - cy) * z / fy
z = depth
```

**Normalization:**
1. Center at origin: `points -= centroid`
2. Scale to unit sphere: `points /= max_distance`

**Sampling:**
- If N > 1024: random sampling without replacement
- If N < 1024: oversampling with replacement
- If N = 0: return zeros (fallback)

### Model Architecture

**Input**: [B, 3, 1024] point cloud + [B, 15] vector

**Forward pass:**
1. Input transform (3x3 T-Net)
2. MLP 1: 3→64→64
3. Feature transform (64x64 T-Net)
4. MLP 2: 64→64→128→1024
5. Max pooling: [B, 1024, N] → [B, 1024]
6. Vector fusion: concat with [B, 128]
7. Classification: [B, 1152] → [B, 512] → [B, 256] → [B, num_classes]

**Output**: [B, num_classes] logits

---

## Known Issues & Fixes

### Issue 1: Class Label Indexing ✅ FIXED

**Problem**: `IndexError: Target 8 is out of bounds`

**Cause**: Dataset uses 1-indexed class IDs (1-8), but PyTorch expects 0-indexed (0-7)

**Fix**: Convert class_id to 0-indexed in dataset generator:
```python
target_class = entry['category_id'] - 1
```

**Status**: Fixed and dataset regenerated

---

## Usage

### Generate Dataset

```bash
# Requires existing RGB-D dataset
uv run python src/vision/dataset_pointnet.py
```

### Train Model

```bash
# Edit config.py: CNN_ARCHITECTURE = "PointNet"
uv run python src/vision/train_pointnet.py
```

### Test Model

```bash
# Test on 10 random samples
uv run python scripts/test_pointnet.py

# Test with visualization
uv run python scripts/test_pointnet.py --visualize

# Test on 50 samples
uv run python scripts/test_pointnet.py --num_samples 50
```

---

## Expected Results

### Training Performance

- **Training time**: 10-20 minutes (CPU), 2-5 minutes (GPU)
- **Convergence**: ~20-30 epochs
- **Final accuracy**: 85-95% (validation)

### Test Performance

- **Inference time**: ~10ms per sample (CPU)
- **Multi-view accuracy**: 90-98% (with 5+ views)
- **Single-view accuracy**: 80-90%

---

## Integration with NBV Environment

### Current Status

PointNet is trained as a standalone classifier. To integrate with NBV environment:

### Required Changes

1. **Update observation space** in `environment.py`:
   ```python
   self.observation_space = spaces.Dict({
       "points": spaces.Box(low=-1, high=1, shape=(3, 1024), dtype=np.float32),
       "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
   })
   ```

2. **Convert depth to point cloud** in `_get_obs()`:
   ```python
   from src.vision.dataset_pointnet import depth_to_point_cloud, normalize_point_cloud, sample_point_cloud
   
   points = depth_to_point_cloud(depth, intrinsics)
   points = normalize_point_cloud(points)
   points = sample_point_cloud(points, 1024)
   ```

3. **Update vision model loading**:
   ```python
   # In config.py
   CNN_ARCHITECTURE = "PointNet"
   ```

---

## Advantages Over CNN

### Pros

1. **Geometric understanding**: Direct 3D reasoning
2. **Rotation invariance**: Built-in via T-Net
3. **Occlusion robustness**: Point clouds handle partial views naturally
4. **Efficient**: Fewer parameters than image CNNs

### Cons

1. **No texture**: Loses RGB color information
2. **Sampling artifacts**: Fixed point count may lose detail
3. **Depth dependency**: Requires accurate depth maps

---

## Future Improvements

### Short-term

1. **Data augmentation**:
   - Random rotation
   - Random jitter (Gaussian noise)
   - Random scaling
   - Point dropout

2. **Multi-view fusion**:
   - Aggregate features from multiple views
   - Attention-based view selection

### Long-term

1. **PointNet++ integration**:
   - Hierarchical feature learning
   - Multi-scale grouping
   - Better local feature capture

2. **RGB-D fusion**:
   - Combine point cloud with RGB features
   - Best of both worlds

---

## Files Modified/Created

### Created Files

1. `src/vision/models.py` - Added `TNet` and `PointNet` classes
2. `src/vision/dataset_pointnet.py` - Dataset generator
3. `src/vision/train_pointnet.py` - Training script
4. `scripts/test_pointnet.py` - Test script
5. `POINTNET_GUIDE.md` - User guide
6. `POINTNET_SUMMARY.md` - This file

### Modified Files

1. `config.py` - Added PointNet parameters
2. `src/vision/models.py` - Updated `NetLoader`

---

## Commands Reference

```bash
# Full pipeline
uv run python src/vision/dataset.py              # Generate RGB-D dataset
uv run python src/vision/dataset_pointnet.py     # Convert to point clouds
uv run python src/vision/train_pointnet.py       # Train PointNet
uv run python scripts/test_pointnet.py           # Test model

# With visualization
uv run python scripts/test_pointnet.py --visualize
```

---

## Conclusion

PointNet implementation is complete and ready for use. The model provides a strong baseline for 3D point cloud classification and can be integrated into the NBV environment for geometric reasoning.

**Next steps:**
1. Wait for training to complete
2. Evaluate performance on test set
3. Compare with MultiModalNet baseline
4. Integrate into NBV environment (optional)

---

**Last Updated**: 2026-04-22
**Status**: Implementation Complete, Training In Progress
