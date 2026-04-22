# Changelog

## 2026-04-22 - Critical Dataset Generation Fixes

### Fixed Issues

1. **Stage 2 incorrectly generated obstacles**
   - Before: 1 object + obstacles
   - After: 2-10 objects, NO obstacles ✅

2. **Stage 3 only generated 1 object**
   - Before: 1 object + obstacles
   - After: 2-10 objects + 1-5 obstacles ✅

3. **Visualization duplicated objects 5x**
   - Before: Used hardcoded camera target `[0.5, 0.0, 0.2]`
   - After: Uses saved `target` and `up` from cameras.json ✅

### Technical Changes

**New Files:**
- `src/vision/dataset_stage.py` - Stage-aware dataset generator
- `FIXES_APPLIED.md` - Detailed fix documentation
- `TRAINING_STAGES_GUIDE.md` - Complete guide for all 3 stages
- `CHANGELOG.md` - This file

**Modified Files:**
- `src/simulation/asset_loader.py` - Added class tracking
- `scripts/generate_sample_viewer.py` - Fixed camera transform
- `scripts/prepare_stage_datasets.py` - Use new generator

**Deprecated Files:**
- `src/vision/dataset.py` - Old single-object generator (kept for reference)

### Verification

All 3 stages regenerated and verified:

**Stage 1 (sample_00000):**
- 1 target object ✅
- 0 obstacles ✅
- 1 robot ✅

**Stage 2 (sample_00000):**
- 9 target objects ✅
- 0 obstacles ✅
- 1 robot ✅

**Stage 3 (sample_00000):**
- 2 target objects ✅
- 2 obstacles ✅
- 1 robot ✅

### Next Steps

1. Generate full datasets (1000 samples per class):
   ```bash
   uv run python scripts/master_pipeline.py --samples 8000
   ```

2. Update Kaggle datasets:
   - Stage 1: https://www.kaggle.com/datasets/sergeykurchev/nbv-stage1-dataset
   - Stage 2: https://www.kaggle.com/datasets/sergeykurchev/nbv-stage2-dataset
   - Stage 3: https://www.kaggle.com/datasets/sergeykurchev/nbv-stage3-dataset

3. Train models on corrected datasets

---

**Status**: All critical issues resolved ✅
