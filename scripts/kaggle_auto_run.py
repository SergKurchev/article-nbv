"""
Push notebook to Kaggle with retry logic and auto-run.
"""

import json
import subprocess
import time
import sys
from pathlib import Path

def run_command(cmd, description="", max_retries=3):
    """Run command with retry logic."""
    for attempt in range(max_retries):
        try:
            print(f"\n{description} (attempt {attempt + 1}/{max_retries})")
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.stdout:
                print(result.stdout)
            if result.stderr and result.returncode != 0:
                print(f"Error: {result.stderr}")

            if result.returncode == 0:
                return result

            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        except subprocess.TimeoutExpired:
            print(f"Command timed out")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(5)
        except Exception as e:
            print(f"Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)

    raise RuntimeError(f"Failed after {max_retries} attempts")

def check_credentials():
    """Check if Kaggle credentials are configured."""
    print("Checking Kaggle credentials...")
    try:
        result = subprocess.run(
            ["kaggle", "config", "view"],
            capture_output=True,
            text=True,
            timeout=10
        )
        print("[OK] Kaggle credentials found")
        return True
    except:
        print("[ERROR] Kaggle credentials not found")
        return False

def push_kernel():
    """Push kernel to Kaggle."""
    print("\n" + "="*60)
    print("Pushing kernel to Kaggle...")
    print("="*60)

    # Change to notebooks directory
    notebooks_dir = Path("notebooks")

    # Push kernel
    result = run_command(
        f'cd "{notebooks_dir}" && kaggle kernels push',
        "Pushing kernel",
        max_retries=3
    )

    print("\n[OK] Kernel pushed successfully!")
    return True

def get_kernel_status(kernel_slug):
    """Get kernel execution status."""
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "status", kernel_slug],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout
    except:
        return None

def wait_for_kernel_start(kernel_slug, max_wait=300):
    """Wait for kernel to start running."""
    print(f"\nWaiting for kernel to start (max {max_wait}s)...")

    start_time = time.time()
    while time.time() - start_time < max_wait:
        status = get_kernel_status(kernel_slug)
        if status:
            print(f"Status: {status.strip()}")

            if "running" in status.lower():
                print("\n[OK] Kernel is running!")
                return True
            elif "complete" in status.lower():
                print("\n[OK] Kernel completed!")
                return True
            elif "error" in status.lower():
                print("\n[ERROR] Kernel failed!")
                return False

        time.sleep(10)

    print("\n[WARNING] Timeout waiting for kernel to start")
    return False

def main():
    print("="*60)
    print("Kaggle Kernel Auto-Push & Run")
    print("="*60)

    # Check credentials
    if not check_credentials():
        print("\nError: Kaggle credentials not configured")
        print("Run: kaggle config set -n username -v YOUR_USERNAME")
        print("Run: kaggle config set -n key -v YOUR_API_KEY")
        sys.exit(1)

    # Push kernel
    try:
        push_kernel()
    except Exception as e:
        print(f"\n[ERROR] Failed to push kernel: {e}")
        sys.exit(1)

    # Kernel slug
    kernel_slug = "sergeykurchev/nbv-training-pointnet-rl"

    # Wait for kernel to start
    print("\n" + "="*60)
    print("Monitoring kernel execution...")
    print("="*60)

    success = wait_for_kernel_start(kernel_slug, max_wait=300)

    if success:
        print("\n" + "="*60)
        print("SUCCESS! Kernel is running on Kaggle")
        print("="*60)
        print(f"\nView kernel: https://www.kaggle.com/code/{kernel_slug}")
        print("\nThe kernel will:")
        print("  1. Download Stage 3 dataset (~5 min)")
        print("  2. Generate PointNet dataset (~10 min)")
        print("  3. Train PointNet (~1-2 hours)")
        print("  4. Train RL agent (~2-4 hours)")
        print("  5. Package results for download")
        print("\nTotal time: ~4-7 hours")
        print("\nDownload results from Output tab when complete.")
    else:
        print("\n" + "="*60)
        print("WARNING: Could not verify kernel started")
        print("="*60)
        print(f"\nManually check: https://www.kaggle.com/code/{kernel_slug}")
        print("Click 'Run All' if kernel didn't auto-start")

    print("="*60)

if __name__ == "__main__":
    main()
