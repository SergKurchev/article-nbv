#!/usr/bin/env python
"""Upload remaining stages 2 and 3 to Kaggle."""
import subprocess
import sys
from datetime import datetime

def run_upload(stage):
    """Run upload for a specific stage."""
    print(f"\n{'='*60}")
    print(f"Starting Stage {stage} upload at {datetime.now()}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [sys.executable, "scripts/update_kaggle_datasets.py", "--stage", str(stage), "--yes"],
        cwd=".",
        stdout=sys.stdout,
        stderr=sys.stderr
    )

    if result.returncode == 0:
        print(f"\n[OK] Stage {stage} completed successfully!")
    else:
        print(f"\n[ERROR] Stage {stage} failed with code {result.returncode}")

    return result.returncode == 0

if __name__ == "__main__":
    print("="*60)
    print("Uploading Stages 2 and 3 to Kaggle")
    print(f"Started at {datetime.now()}")
    print("="*60)

    # Upload Stage 2
    success_2 = run_upload(2)

    # Upload Stage 3
    success_3 = run_upload(3)

    print("\n" + "="*60)
    print("Upload Summary")
    print("="*60)
    print(f"Stage 2: {'SUCCESS' if success_2 else 'FAILED'}")
    print(f"Stage 3: {'SUCCESS' if success_3 else 'FAILED'}")
    print(f"Completed at {datetime.now()}")
    print("="*60)
