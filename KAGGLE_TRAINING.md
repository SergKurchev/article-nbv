# Kaggle Training Guide

## Quick Start

### 1. Upload to Kaggle

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Click "File" → "Upload Notebook"
4. Upload `notebooks/kaggle_training.ipynb`

### 2. Add Dataset Inputs

In the notebook settings, add dataset:
- **Stage 3**: `sergeykurchev/nbv-stage3-dataset`

Or add as Kaggle input in the right panel.

### 3. Enable GPU

- Click "Settings" (right panel)
- Accelerator: **GPU T4 x2** (recommended)
- Internet: **On** (for downloading packages)

### 4. Run Notebook

Click "Run All" or run cells sequentially.

## Training Configuration

### Current Settings (in notebook)
- **Dataset**: Stage 3 only (most complex)
- **PointNet epochs**: 50
- **RL timesteps**: 50,000 (reduced from 500k)
- **Object mode**: primitives

### Estimated Time
- Dataset download: ~5 min
- PointNet dataset generation: ~10 min
- PointNet training: ~1-2 hours (GPU)
- RL training: ~2-4 hours (GPU)
- **Total**: ~4-7 hours

## Output Files

After training completes, download `kaggle_results.zip` containing:

```
kaggle_results/
├── weights/
│   ├── pointnet_best.pt          # Best PointNet model
│   ├── pointnet_last.pt           # Last checkpoint
│   └── pointnet_metrics.csv       # Training metrics
├── rl_run/
│   ├── best_policy.zip            # Best RL policy
│   ├── last_policy.zip            # Last checkpoint
│   ├── rl_metrics.csv             # RL training metrics
│   └── train_cnn_probs.csv        # CNN probability logs
└── SUMMARY.txt                    # Training summary
```

## CSV Metrics Format

### pointnet_metrics.csv
```csv
epoch,train_loss,train_acc,val_loss,val_acc,learning_rate
1,2.0543,45.23,1.8932,52.10,0.001
2,1.7821,58.67,1.6543,61.45,0.001
...
```

### rl_metrics.csv
```csv
step,reward,obj_metric,reward_std,obj_metric_std
2000,12.45,0.23,3.21,0.05
4000,15.67,0.31,2.89,0.04
...
```

## Troubleshooting

### "Dataset not found"
- Make sure dataset is added as Kaggle input
- Check dataset slug: `sergeykurchev/nbv-stage3-dataset`

### "Out of memory"
- Reduce batch size in config
- Use smaller dataset subset

### "Timeout"
- Kaggle has 9-12 hour limit
- RL training may timeout - partial results still saved
- Download intermediate checkpoints

### "uv not found"
- Notebook installs uv automatically
- If fails, fallback to pip

## Continue Training Locally

1. Download `kaggle_results.zip`
2. Extract to project directory
3. Copy weights and runs:
   ```bash
   cp -r kaggle_results/weights/* weights/primitives/
   cp -r kaggle_results/rl_run runs/
   ```
4. Resume training:
   ```bash
   uv run python train.py --load best
   ```

## Customization

### Change RL timesteps
Edit in notebook cell 1:
```python
RL_TIMESTEPS = 100000  # Increase if time allows
```

### Use different dataset stage
```python
DATASET_STAGE = 2  # Stage 1, 2, or 3
```

### Change PointNet epochs
```python
POINTNET_EPOCHS = 100  # More epochs for better accuracy
```

## Repository Setup

**Important**: Update repository URL in notebook!

Edit cell 1:
```python
REPO_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO.git"
```

Or upload entire project as Kaggle dataset instead of cloning.

## Notes

- Training uses Stage 3 dataset (objects + obstacles) for most realistic scenarios
- RL timesteps reduced to 50k to fit Kaggle time limits
- All metrics logged to CSV for easy analysis
- Results automatically packaged for download
- Can resume training locally with downloaded weights
