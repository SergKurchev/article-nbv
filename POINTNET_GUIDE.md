# PointNet Integration Guide

## Overview

PointNet is a deep learning architecture for processing 3D point clouds. This implementation converts RGB-D images from the existing dataset into point clouds and trains a PointNet classifier.

---

## Architecture

### PointNet Components

1. **Input Transform (T-Net)**
   - Learns 3x3 transformation matrix
   - Provides rotation invariance
   - Aligns input point cloud

2. **Feature Transform (T-Net)**
   - Learns 64x64 transformation matrix
   - Aligns feature space
   - Improves feature learning

3. **Shared MLP Layers**
   - Conv1D layers: 3→64→64→64→128→1024
   - BatchNorm + ReLU activation
   - Per-point feature extraction

4. **Symmetric Max Pooling**
   - Global max pooling across all points
   - Provides permutation invariance
   - Aggregates point features into global descriptor

5. **Vector Fusion (Optional)**
   - Fuses robot state (15D vector) with point features
   - MLP encoder: 15→64→128
   - Concatenates with global descriptor

6. **Classification Head**
   - Fully connected layers: 1024+128→512→256→num_classes
   - BatchNorm + Dropout (0.3)
   - Outputs class logits

### Key Features

- **Permutation Invariance**: Max pooling ensures order-independent processing
- **Rotation Invariance**: Input transform aligns point clouds
- **Multi-modal**: Fuses point cloud with robot state vector
- **Regularization**: Dropout and BatchNorm for generalization

---

## Dataset Generation

### Conversion Pipeline

1. **Depth to Point Cloud**
   - Back-project depth map to 3D using camera intrinsics
   - Filter invalid depths (depth > 0 and depth < 2.0m)
   - Convert pixel coordinates to 3D world coordinates

2. **Normalization**
   - Center point cloud at origin (subtract centroid)
   - Scale to unit sphere (divide by max distance)

3. **Sampling**
   - Sample fixed number of points (default: 1024)
   - Random sampling if N > 1024
   - Oversampling with replacement if N < 1024

4. **Vector Extraction**
   - Extract camera pose: [x, y, z, qx, qy, qz, qw]
   - Pad to 15D to match MultiModalNet format

### Dataset Structure

```
dataset_pointnet/primitives/
├── sample_00000/
│   ├── point_clouds.npy    # [num_views, 1024, 3] point clouds
│   ├── vectors.npy         # [num_views, 15] camera poses
│   └── metadata.json       # class_id, num_views, num_points
├── sample_00001/
└── ...
```

### Generation Command

```bash
# Generate PointNet dataset from existing RGB-D dataset
uv run python src/vision/dataset_pointnet.py
```

**Requirements:**
- Existing RGB-D dataset in `dataset/primitives/`
- Run `uv run python src/vision/dataset.py` first if needed

---

## Training

### Training Command

```bash
# Train PointNet on point cloud dataset
uv run python src/vision/train_pointnet.py
```

### Configuration

Edit `config.py`:

```python
# PointNet parameters
POINTNET_NUM_POINTS = 1024  # Points per sample
POINTNET_BATCH_SIZE = 32    # Batch size
POINTNET_LR = 1e-3          # Learning rate
POINTNET_EPOCHS = 50        # Training epochs

# Architecture selection
CNN_ARCHITECTURE = "PointNet"
```

### Training Process

1. **Data Loading**
   - Load point clouds and vectors from dataset
   - Split into train/val (80/20)
   - Random view selection per sample

2. **Optimization**
   - Loss: CrossEntropyLoss
   - Optimizer: Adam (lr=1e-3)
   - Scheduler: StepLR (step_size=20, gamma=0.5)

3. **Checkpointing**
   - Save best model based on validation accuracy
   - Save last model after each epoch
   - Output: `weights/primitives/pointnet_best.pt`

### Expected Performance

- **Training time**: ~10-20 minutes (CPU), ~2-5 minutes (GPU)
- **Accuracy**: 85-95% on validation set (8 primitive classes)
- **Parameters**: ~3.5M trainable parameters

---

## Usage in NBV Environment

### Loading PointNet Model

```python
from src.vision.models import PointNet
import torch

# Create model
model = PointNet(
    num_classes=8,
    vector_dim=15,
    num_points=1024,
    use_vector=True
)

# Load weights
model.load_state_dict(torch.load('weights/primitives/pointnet_best.pt'))
model.eval()
```

### Inference

```python
# Convert depth map to point cloud
from src.vision.dataset_pointnet import depth_to_point_cloud, sample_point_cloud, normalize_point_cloud

points = depth_to_point_cloud(depth, intrinsics)
points = normalize_point_cloud(points)
points = sample_point_cloud(points, num_points=1024)

# Prepare input [1, 3, 1024]
points = torch.from_numpy(points.T).unsqueeze(0).float()
vector = torch.from_numpy(camera_pose).unsqueeze(0).float()

# Forward pass
with torch.no_grad():
    logits = model(points, vector)
    probs = torch.softmax(logits, dim=1)
```

### Integration with Environment

To use PointNet in the NBV environment:

1. **Update config.py**:
   ```python
   CNN_ARCHITECTURE = "PointNet"
   ```

2. **Modify environment observation**:
   - Convert depth map to point cloud in `_get_obs()`
   - Replace RGB-D image with point cloud [3, 1024]

3. **Update observation space**:
   ```python
   self.observation_space = spaces.Dict({
       "points": spaces.Box(low=-1, high=1, shape=(3, 1024), dtype=np.float32),
       "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
   })
   ```

---

## Advantages of PointNet

### Compared to CNN (MultiModalNet)

**Pros:**
- **Geometric reasoning**: Direct 3D spatial understanding
- **Rotation invariance**: T-Net provides built-in rotation handling
- **Occlusion robustness**: Point clouds naturally handle partial views
- **Efficient**: Fewer parameters than image-based CNNs

**Cons:**
- **No texture**: Loses RGB color information
- **Sampling artifacts**: Fixed point count may lose detail
- **Depth dependency**: Requires accurate depth maps

### Use Cases

- **Best for**: Geometric classification, shape recognition, occlusion handling
- **Not ideal for**: Texture-based classification (red/green gradients)
- **Recommended**: Combine with RGB features for best results

---

## Troubleshooting

### Issue: Empty Point Clouds

**Symptom**: Point clouds have 0 points after conversion

**Solution:**
- Check depth map validity (depth > 0)
- Verify camera intrinsics are correct
- Increase max_depth threshold (default: 2.0m)

### Issue: Low Accuracy

**Symptom**: Validation accuracy < 70%

**Solution:**
- Increase training epochs (50→100)
- Increase number of points (1024→2048)
- Add data augmentation (rotation, jitter)
- Check point cloud normalization

### Issue: Out of Memory

**Symptom**: CUDA out of memory during training

**Solution:**
- Reduce batch size (32→16)
- Reduce number of points (1024→512)
- Use CPU training (slower but stable)

---

## Future Improvements

### PointNet++ Integration

PointNet++ adds hierarchical feature learning:
- Multi-scale grouping
- Set abstraction layers
- Better local feature capture

### Data Augmentation

Add augmentation for robustness:
- Random rotation
- Random jitter (Gaussian noise)
- Random scaling
- Random point dropout

### Multi-View Fusion

Aggregate features from multiple views:
- Attention-based view fusion
- Temporal consistency
- View selection policy

---

## References

- **PointNet Paper**: [Qi et al., CVPR 2017](https://arxiv.org/abs/1612.00593)
- **PointNet++ Paper**: [Qi et al., NeurIPS 2017](https://arxiv.org/abs/1706.02413)
- **Official Implementation**: [charlesq34/pointnet](https://github.com/charlesq34/pointnet)

---

## Commands Summary

```bash
# 1. Generate RGB-D dataset (if not exists)
uv run python src/vision/dataset.py

# 2. Convert to PointNet format
uv run python src/vision/dataset_pointnet.py

# 3. Train PointNet
uv run python src/vision/train_pointnet.py

# 4. Test PointNet (TODO: create test script)
uv run python scripts/test_pointnet.py
```

---

**Last Updated**: 2026-04-22
**Version**: 1.0
