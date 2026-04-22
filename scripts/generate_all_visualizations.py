"""
Generate HTML visualizations for all samples in all 3 stages.

Usage:
    python scripts/generate_all_visualizations.py
    python scripts/generate_all_visualizations.py --stage 1
    python scripts/generate_all_visualizations.py --max-samples 3
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import subprocess
from tqdm import tqdm
import config

def generate_visualizations(stage=None, max_samples=3, stride=2, max_points=500000):
    """Generate visualizations for all samples in specified stages."""

    stages = [stage] if stage else [1, 2, 3]

    total_generated = 0
    total_failed = 0

    for s in stages:
        dataset_dir = config.BASE_DIR / "dataset" / config.OBJECT_MODE / f"stage{s}"

        if not dataset_dir.exists():
            print(f"\n[WARNING] Stage {s} dataset not found: {dataset_dir}")
            continue

        samples = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")])

        # Always limit to first 3 samples by default
        samples = samples[:max_samples]

        print(f"\n{'='*60}")
        print(f"Stage {s}: Processing {len(samples)} samples")
        print(f"{'='*60}")

        for sample_dir in tqdm(samples, desc=f"Stage {s}"):
            try:
                # Run viewer generator
                result = subprocess.run(
                    ["uv", "run", "python", "scripts/generate_sample_viewer.py",
                     str(sample_dir),
                     "--stride", str(stride),
                     "--max-points", str(max_points)],
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if result.returncode == 0:
                    total_generated += 1
                else:
                    print(f"\n[ERROR] Failed for {sample_dir.name}:")
                    print(result.stderr)
                    total_failed += 1

            except subprocess.TimeoutExpired:
                print(f"\n[ERROR] Timeout for {sample_dir.name}")
                total_failed += 1
            except Exception as e:
                print(f"\n[ERROR] Exception for {sample_dir.name}: {e}")
                total_failed += 1

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Generated: {total_generated}")
    print(f"Failed: {total_failed}")
    print(f"\nVisualization files: */visualization.html")
    print("Open any .html file in your browser to view the 3D point cloud")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], help="Generate for specific stage only")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples per stage")
    parser.add_argument("--stride", type=int, default=2, help="Pixel stride (default: 2)")
    parser.add_argument("--max-points", type=int, default=500000, help="Max points per sample (default: 500000)")
    args = parser.parse_args()

    generate_visualizations(
        stage=args.stage,
        max_samples=args.max_samples,
        stride=args.stride,
        max_points=args.max_points
    )
