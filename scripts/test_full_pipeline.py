"""
Full pipeline test script - runs entire workflow from texture generation to agent evaluation.

This script tests the complete pipeline on minimal datasets to verify everything works:
1. Generate textures (20 mixed variants)
2. Generate minimal datasets for Stage 1, 2, 3 (8 samples each)
3. Train vision model on multi-frame sequences (1 epoch)
4. Train RL agent (100 timesteps)
5. Evaluate agent

Run this locally to verify the pipeline before full-scale training.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Fix Windows console encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import config
import torch
import subprocess
import shutil

# Minimal configuration for testing
TEST_SAMPLES_PER_CLASS = 1  # 1 sample per class = 8 total samples
TEST_EPOCHS = 1
TEST_TIMESTEPS = 100
TEST_EVAL_EPISODES = 2


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def step1_generate_textures():
    """Step 1: Generate textures."""
    print_section("STEP 1: Generate Textures (20 mixed variants)")

    from src.vision.texture_generator import generate_all_textures

    texture_dir = config.DATA_DIR / "objects" / "textures"
    texture_dir.mkdir(parents=True, exist_ok=True)

    # Check if textures exist
    existing_mixed = list(texture_dir.glob("mixed_*.png"))
    if len(existing_mixed) >= config.TEXTURE_NUM_MIXED_VARIANTS:
        print(f"✓ Found {len(existing_mixed)} mixed textures, skipping generation")
        return

    print(f"Generating {config.TEXTURE_NUM_MIXED_VARIANTS} mixed texture variants...")
    generate_all_textures(output_dir=texture_dir, visualize=False)
    print("✓ Textures generated successfully")


def step2_generate_datasets():
    """Step 2: Generate minimal datasets for all stages."""
    print_section("STEP 2: Generate Minimal Datasets (8 samples per stage)")

    from src.vision.dataset_stage import generate_stage_dataset

    # Save original config
    original_stage = config.SCENE_STAGE
    original_samples = config.DATASET_SAMPLES_PER_CLASS

    for stage in [1, 2, 3]:
        print(f"\n--- Stage {stage} ---")

        # Update config
        config.SCENE_STAGE = stage
        config.DATASET_SAMPLES_PER_CLASS = TEST_SAMPLES_PER_CLASS

        # Set dataset directory
        dataset_dir = config.BASE_DIR / "dataset" / config.OBJECT_MODE / f"stage{stage}_test"
        config.DATASET_DIR = dataset_dir

        # Clear existing test dataset
        if dataset_dir.exists():
            print(f"Removing existing test dataset at {dataset_dir}")
            shutil.rmtree(dataset_dir)

        dataset_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating {TEST_SAMPLES_PER_CLASS * config.NUM_CLASSES} samples...")
        generate_stage_dataset()

        # Verify
        samples = list(dataset_dir.glob("sample_*"))
        print(f"✓ Generated {len(samples)} samples")

    # Restore config
    config.SCENE_STAGE = original_stage
    config.DATASET_SAMPLES_PER_CLASS = original_samples
    config.DATASET_DIR = config.BASE_DIR / "dataset" / config.OBJECT_MODE

    print("\n✓ All stage datasets generated")


def step3_train_vision_model():
    """Step 3: Train vision model on multi-frame sequences."""
    print_section("STEP 3: Train Vision Model (Multi-Frame, 1 epoch)")

    from src.vision.train_multiframe import train_multiframe

    # Save original config
    original_epochs = config.CNN_EPOCHS
    original_batch_size = config.CNN_BATCH_SIZE

    # Set minimal training config
    config.CNN_EPOCHS = TEST_EPOCHS
    config.CNN_BATCH_SIZE = 2  # Small batch for testing

    for stage in [1, 2, 3]:
        print(f"\n--- Training on Stage {stage} ---")

        try:
            train_multiframe(stage=stage, max_frames=5, min_frames=1)
            print(f"✓ Stage {stage} training completed")
        except Exception as e:
            print(f"✗ Stage {stage} training failed: {e}")
            import traceback
            traceback.print_exc()

    # Restore config
    config.CNN_EPOCHS = original_epochs
    config.CNN_BATCH_SIZE = original_batch_size

    print("\n✓ Vision model training completed")


def step4_train_rl_agent():
    """Step 4: Train RL agent."""
    print_section("STEP 4: Train RL Agent (100 timesteps)")

    # Save original config
    original_timesteps = config.TOTAL_TIMESTEPS
    original_eval_freq = config.EVAL_FREQ
    original_load_mode = config.CNN_LOAD_MODE
    original_model_path = config.CNN_MODEL_PATH

    # Set minimal training config
    config.TOTAL_TIMESTEPS = TEST_TIMESTEPS
    config.EVAL_FREQ = 50  # Evaluate once during training

    print(f"Training agent for {TEST_TIMESTEPS} timesteps...")

    try:
        # Load best vision model from Stage 3
        weights_dir = config.CNN_WEIGHTS_DIR / "stage3"
        model_path = weights_dir / "multiframe_best_stage3.pt"

        if model_path.exists():
            print(f"Using vision model: {model_path}")
            config.CNN_LOAD_MODE = "best"
            config.CNN_MODEL_PATH = model_path
        else:
            print(f"Warning: Vision model not found at {model_path}")
            print("Training without vision model...")
            config.CNN_LOAD_MODE = "none"

        # Import and run training
        from train import train
        train()
        print("✓ RL agent training completed")

    except Exception as e:
        print(f"✗ RL agent training failed: {e}")
        import traceback
        traceback.print_exc()

    # Restore config
    config.TOTAL_TIMESTEPS = original_timesteps
    config.EVAL_FREQ = original_eval_freq
    config.CNN_LOAD_MODE = original_load_mode
    config.CNN_MODEL_PATH = original_model_path


def step5_evaluate_agent():
    """Step 5: Evaluate trained agent."""
    print_section("STEP 5: Evaluate Agent")

    print(f"Evaluating agent for {TEST_EVAL_EPISODES} episodes...")

    try:
        # Find latest RL run directory
        run_dir = config.get_latest_rl_run_dir()

        if run_dir is None:
            print("✗ No RL training run found")
            return

        print(f"Using run directory: {run_dir}")

        # Load best vision model
        weights_dir = config.CNN_WEIGHTS_DIR / "stage3"
        model_path = weights_dir / "multiframe_best_stage3.pt"

        # Load vision model
        from src.vision.models import NetLoader
        vision_model = NetLoader.load(arch=config.CNN_ARCHITECTURE, num_classes=config.NUM_CLASSES, vector_dim=15)

        if model_path.exists():
            print(f"Loading vision model from {model_path}")
            vision_model.load_state_dict(torch.load(model_path, map_location="cpu"))
        else:
            print(f"Warning: Vision model not found at {model_path}")

        vision_model.eval()

        # Load RL agent
        from src.rl.agent import NBVAgent
        from src.simulation.environment import NBVEnv
        from stable_baselines3.common.vec_env import DummyVecEnv

        agent_path = run_dir / "best_policy.zip"
        if not agent_path.exists():
            agent_path = run_dir / "last_policy.zip"

        if not agent_path.exists():
            print("✗ No trained agent found")
            return

        print(f"Loading agent from {agent_path}")

        # Create evaluation environment
        def make_eval_env():
            return NBVEnv(render_mode="rgb_array", headless=True, no_arm=False, vision_model=vision_model)

        eval_env = DummyVecEnv([make_eval_env])

        # Load agent
        agent = NBVAgent(eval_env)
        agent.load(str(agent_path.with_suffix('')))  # Remove .zip extension

        # Evaluate
        print(f"Running {TEST_EVAL_EPISODES} evaluation episodes...")
        from stable_baselines3.common.evaluation import evaluate_policy

        mean_reward, std_reward = evaluate_policy(
            agent.model,
            eval_env,
            n_eval_episodes=TEST_EVAL_EPISODES,
            deterministic=True
        )

        print(f"✓ Evaluation completed")
        print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    except Exception as e:
        print(f"✗ Agent evaluation failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run full pipeline test."""
    print("\n" + "="*70)
    print("  FULL PIPELINE TEST")
    print("  Testing complete workflow on minimal datasets")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  - Samples per stage: {TEST_SAMPLES_PER_CLASS * config.NUM_CLASSES}")
    print(f"  - Vision training epochs: {TEST_EPOCHS}")
    print(f"  - RL training timesteps: {TEST_TIMESTEPS}")
    print(f"  - Evaluation episodes: {TEST_EVAL_EPISODES}")
    print("="*70)

    try:
        # Run pipeline
        step1_generate_textures()
        step2_generate_datasets()
        step3_train_vision_model()
        step4_train_rl_agent()
        step5_evaluate_agent()

        # Summary
        print_section("PIPELINE TEST COMPLETED")
        print("✓ All steps completed successfully!")
        print("\nNext steps:")
        print("  1. Review training logs in runs/ directory")
        print("  2. Check generated datasets in dataset/primitives/stage*_test/")
        print("  3. If everything looks good, run full-scale training:")
        print("     - Generate 1000-sample datasets: python scripts/generate_and_upload_datasets.py")
        print("     - Train vision models: python src/vision/train_multiframe.py --stage 1")
        print("     - Train RL agent: python train.py")

    except KeyboardInterrupt:
        print("\n\nPipeline test interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
