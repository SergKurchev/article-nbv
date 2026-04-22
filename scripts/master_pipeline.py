"""
Master script to prepare all stage datasets, test training, and upload to Kaggle.

Usage:
    # Full pipeline with minimal samples (fast test)
    uv run python scripts/master_pipeline.py --samples 8 --test-only

    # Generate full datasets and upload
    uv run python scripts/master_pipeline.py --samples 800 --upload

    # Only upload existing datasets
    uv run python scripts/master_pipeline.py --upload-only
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import subprocess
import argparse

def run_command(cmd, description):
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n[ERROR] Failed: {description}")
        return False

    print(f"\n[OK] Success: {description}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Master pipeline for NBV dataset preparation")
    parser.add_argument("--samples", type=int, default=8, help="Total samples per stage (default: 8 for testing)")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], help="Process specific stage only")
    parser.add_argument("--test-only", action="store_true", help="Only test training, don't upload")
    parser.add_argument("--upload-only", action="store_true", help="Only upload existing datasets")
    parser.add_argument("--skip-test", action="store_true", help="Skip training test")
    parser.add_argument("--kaggle-username", type=str, default="sergeykurchev", help="Kaggle username")
    parser.add_argument("--kaggle-key", type=str, default="fd9ae7ea316d408e492e260be6c3727e", help="Kaggle API key")
    args = parser.parse_args()

    stages = [args.stage] if args.stage else [1, 2, 3]

    print(f"""
{'='*60}
NBV Dataset Preparation Pipeline
{'='*60}
Stages: {stages}
Samples per stage: {args.samples}
Test training: {not args.skip_test and not args.upload_only}
Upload to Kaggle: {not args.test_only}
{'='*60}
""")

    # Step 1: Generate datasets (unless upload-only)
    if not args.upload_only:
        for stage in stages:
            success = run_command(
                ["uv", "run", "python", "scripts/prepare_stage_datasets.py",
                 "--stage", str(stage),
                 "--samples", str(args.samples)],
                f"Generate Stage {stage} Dataset ({args.samples} samples)"
            )
            if not success:
                print(f"\n[ERROR] Pipeline failed at dataset generation for Stage {stage}")
                return 1

    # Step 2: Test training (unless skipped or upload-only)
    if not args.skip_test and not args.upload_only:
        for stage in stages:
            success = run_command(
                ["uv", "run", "python", "scripts/test_training_single_sample.py",
                 "--stage", str(stage)],
                f"Test Training on Stage {stage} Dataset"
            )
            if not success:
                print(f"\n[ERROR] Pipeline failed at training test for Stage {stage}")
                return 1

    # Step 3: Upload to Kaggle (unless test-only)
    if not args.test_only:
        for stage in stages:
            success = run_command(
                ["uv", "run", "python", "scripts/upload_to_kaggle.py",
                 "--stage", str(stage),
                 "--username", args.kaggle_username,
                 "--key", args.kaggle_key],
                f"Upload Stage {stage} Dataset to Kaggle"
            )
            if not success:
                print(f"\n[WARNING] Upload failed for Stage {stage}, but continuing...")

    # Summary
    print(f"\n{'='*60}")
    print("Pipeline Complete!")
    print(f"{'='*60}")
    print("\nDatasets created:")
    for stage in stages:
        dataset_dir = Path("dataset/primitives") / f"stage{stage}"
        if dataset_dir.exists():
            sample_count = len(list(dataset_dir.glob("sample_*")))
            print(f"  Stage {stage}: {sample_count} samples at {dataset_dir}")

    if not args.test_only:
        print("\nKaggle datasets:")
        for stage in stages:
            print(f"  Stage {stage}: https://www.kaggle.com/datasets/{args.kaggle_username}/nbv-stage{stage}-dataset")

    return 0

if __name__ == "__main__":
    sys.exit(main())
