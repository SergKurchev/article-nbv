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

def generate_visualizations(stage=None, max_samples=3, stride=2, max_points=500000, dataset_dir=None, out_dir=None, python_exe="python"):
    """Generate visualizations for all samples in specified stages."""

    stages = [stage] if stage else [1, 2, 3]

    total_generated = 0
    total_failed = 0

    for s in stages:
        if dataset_dir:
            stage_dataset_dir = Path(dataset_dir)
        else:
            stage_dataset_dir = config.BASE_DIR / "dataset" / config.OBJECT_MODE / f"stage{s}"

        if not stage_dataset_dir.exists():
            print(f"\n[WARNING] Stage {s} dataset not found: {stage_dataset_dir}")
            continue

        samples = sorted([d for d in stage_dataset_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")])

        # Always limit to first 3 samples by default
        if max_samples:
            samples = samples[:max_samples]

        print(f"\n{'='*60}")
        print(f"Stage {s}: Processing {len(samples)} samples")
        print(f"{'='*60}")

        for sample_dir in tqdm(samples, desc=f"Stage {s}"):
            try:
                cmd = [
                    python_exe, "scripts/generate_sample_viewer.py",
                    str(sample_dir),
                    "--stride", str(stride),
                    "--max-points", str(max_points)
                ]
                if out_dir:
                    out_path = Path(out_dir) / f"stage{s}" / f"{sample_dir.name}_visualization.html"
                    cmd.extend(["--output", str(out_path)])

                # Run viewer generator
                result = subprocess.run(
                    cmd,
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
    if out_dir:
        print(f"\nVisualization files saved in: {out_dir}")
    else:
        print(f"\nVisualization files: */visualization.html")
    print("Open any .html file in your browser to view the 3D point cloud")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], help="Generate for specific stage only")
    parser.add_argument("--max-samples", type=int, default=3, help="Limit number of samples per stage (default: 3)")
    parser.add_argument("--stride", type=int, default=2, help="Pixel stride (default: 2)")
    parser.add_argument("--max-points", type=int, default=500000, help="Max points per sample (default: 500000)")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Custom dataset directory path")
    parser.add_argument("--out-dir", type=str, default=None, help="Custom output directory for HTML files")
    parser.add_argument("--python", type=str, default="python", help="Python executable to use (default: python)")
    args = parser.parse_args()

    generate_visualizations(
        stage=args.stage,
        max_samples=args.max_samples,
        stride=args.stride,
        max_points=args.max_points,
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        python_exe=args.python
    )
