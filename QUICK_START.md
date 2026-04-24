# Quick Start Guide

## Full Pipeline Execution

### Step 1: Test Pipeline (5-10 minutes)
```bash
# Test everything works on minimal data
python scripts/test_full_pipeline.py
```

This will:
- Generate 20 mixed textures
- Create 8-sample test datasets for Stage 1, 2, 3
- Train vision model for 1 epoch
- Train RL agent for 100 timesteps
- Evaluate agent

### Step 2: Generate Full Datasets (2-3 hours)
```bash
# Generate 1000-sample datasets and upload to Kaggle
python scripts/generate_and_upload_datasets.py
```

This will:
- Generate 1000 samples per stage (Stage 1, 2, 3)
- Verify data integrity
- Upload to Kaggle as public datasets:
  - `sergeykurchev/strawpick-nbv-stage1-dataset`
  - `sergeykurchev/strawpick-nbv-stage2-dataset`
  - `sergeykurchev/strawpick-nbv-stage3-dataset`

### Step 3: Train Vision Models (1-2 hours per stage)
```bash
# Train on Stage 1 (single object)
python src/vision/train_multiframe.py --stage 1 --max-frames 5 --min-frames 1

# Train on Stage 2 (multi-object)
python src/vision/train_multiframe.py --stage 2 --max-frames 5 --min-frames 1

# Train on Stage 3 (multi-object + obstacles)
python src/vision/train_multiframe.py --stage 3 --max-frames 5 --min-frames 1
```

Models saved to: `weights/primitives/stage{1,2,3}/multiframe_best_stage{1,2,3}.pt`

### Step 4: Train RL Agent (2-4 hours)
```bash
# Update config.py to use Stage 3 model
# Set: CNN_MODEL_PATH = CNN_WEIGHTS_DIR / "stage3" / "multiframe_best_stage3.pt"
# Set: CNN_LOAD_MODE = "best"

# Train agent
python train.py
```

Agent saved to: `runs/rl_train_*/best_policy.zip`

### Step 5: Evaluate Agent
```bash
python train.py --mode eval
```

## Alternative: Download Datasets from Kaggle

If datasets are already uploaded to Kaggle:

```bash
# Install Kaggle CLI
pip install kaggle

# Download datasets
kaggle datasets download -d sergeykurchev/strawpick-nbv-stage1-dataset
kaggle datasets download -d sergeykurchev/strawpick-nbv-stage2-dataset
kaggle datasets download -d sergeykurchev/strawpick-nbv-stage3-dataset

# Unzip to correct locations
unzip strawpick-nbv-stage1-dataset.zip -d dataset/primitives/stage1/
unzip strawpick-nbv-stage2-dataset.zip -d dataset/primitives/stage2/
unzip strawpick-nbv-stage3-dataset.zip -d dataset/primitives/stage3/
```

## Key Features

### Mixed Textures
- 20 pre-generated variants with fixed random seeds
- Randomly selected during dataset generation
- Located in: `src/data/objects/textures/mixed_0.png` through `mixed_19.png`

### Multi-Frame Training
- Model trains on 1-5 frames simultaneously
- Handles sequential observation (agent gets frames one at a time)
- Aggregates predictions across frames

### Progressive Training
- Stage 1: Learn basic object recognition (single object)
- Stage 2: Handle multiple objects and occlusions
- Stage 3: Navigate around obstacles

## Monitoring Training

### Vision Model Training
- Logs: `runs/multiframe_train_stage{1,2,3}_*/training_log.csv`
- Plots: `runs/multiframe_train_stage{1,2,3}_*/learning_curve.png`

### RL Agent Training
- Logs: `runs/rl_train_*/training_log.jsonl`
- Videos: `runs/rl_train_*/videos/`
- Plots: `runs/rl_train_*/learning_curve.png`

## Troubleshooting

### "Texture not found" error
```bash
python src/vision/texture_generator.py
```

### "Dataset directory not found" error
```bash
python scripts/prepare_stage_datasets.py --stage 1 --samples 1000
```

### Out of memory during training
Reduce batch size in `config.py`:
```python
CNN_BATCH_SIZE = 8  # Default: 16
```

### Kaggle upload fails
Check credentials:
```bash
cat ~/.kaggle/kaggle.json
```

Should contain:
```json
{
  "username": "sergeykurchev",
  "key": "fd9ae7ea316d408e492e260be6c3727e"
}
```

## Expected Results

### Vision Model
- Stage 1 accuracy: ~90-95% (single object, easy)
- Stage 2 accuracy: ~80-85% (multiple objects, harder)
- Stage 3 accuracy: ~75-80% (obstacles, hardest)

### RL Agent
- Average reward should increase over time
- Accuracy difference (acc_diff) should improve
- Agent should learn to move camera for better views

## Time Estimates

| Task | Time (CPU) | Time (GPU) |
|------|-----------|-----------|
| Test pipeline | 5-10 min | 3-5 min |
| Generate datasets | 2-3 hours | 2-3 hours |
| Train vision (per stage) | 2-3 hours | 30-60 min |
| Train RL agent | 3-4 hours | 1-2 hours |
| **Total** | **~10-15 hours** | **~5-8 hours** |

## Next Steps After Training

1. **Analyze Results:**
   - Check learning curves
   - Review evaluation metrics
   - Watch agent videos

2. **Iterate:**
   - Adjust hyperparameters if needed
   - Try different model architectures
   - Experiment with reward shaping

3. **Deploy:**
   - Export trained models
   - Test on real robot (if available)
   - Create demo videos

## Support

For issues or questions:
1. Check `IMPLEMENTATION_SUMMARY.md` for detailed documentation
2. Review training logs in `runs/` directory
3. Verify dataset integrity with `scripts/prepare_stage_datasets.py --verify-only`
