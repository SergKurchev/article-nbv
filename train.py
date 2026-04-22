import os
import uuid
import sys
import torch
import signal
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv

import config
from src.simulation.environment import NBVEnv
from src.rl.agent import NBVAgent
from src.rl.callbacks import NBVCallback
from src.utils.logger import Logger
from src.vision.models import NetLoader

def handle_interrupt(sig, frame):
    print("\nInterrupt received. Stopping training safely...")
    # Safe shutdown handled inherently by python exceptions optionally, or just sys.exit
    # Actually, we can raise KeyboardInterrupt and catch it in train()
    raise KeyboardInterrupt

signal.signal(signal.SIGINT, handle_interrupt)

def make_env(vision_model, no_arm=False, headless=True):
    def _init():
        return NBVEnv(render_mode="rgb_array", headless=headless, no_arm=no_arm, vision_model=vision_model)
    return _init

def train():
    args = config.get_args()
    
    # Setup paths
    run_dir = None
    if args.load != "none":
        run_dir = config.get_latest_rl_run_dir()
        if run_dir:
            print(f"Resuming from existing run: {run_dir.name}")
        else:
            print(f"No previous runs found for mode={config.OBJECT_MODE}. Starting fresh.")
    
    if run_dir is None:
        run_dir = config.get_run_dir("rl_train")
        print(f"Starting new run: {run_dir.name}")

    run_id = run_dir.name
    video_dir = run_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Custom JSON Logger
    logger = Logger(run_dir)
    logger.copy_config() 
    
    # Load CNN
    vision_model = NetLoader.load(arch=config.CNN_ARCHITECTURE, num_classes=config.NUM_CLASSES, vector_dim=15)
    if config.CNN_LOAD_MODE == "best" and config.CNN_MODEL_PATH.exists():
        print(f"Loading vision weights from {config.CNN_MODEL_PATH}")
        vision_model.load_state_dict(torch.load(config.CNN_MODEL_PATH, map_location="cpu"))
    vision_model.eval()
    
    # Create env
    env = DummyVecEnv([make_env(vision_model, args.no_arm, not args.gui)])
    eval_env = DummyVecEnv([make_env(vision_model, args.no_arm, True)])
    
    agent = NBVAgent(env)
    
    # Load if specified
    if args.load != "none" and run_dir is not None:
        model_name = "best_policy" if args.load == "best" else "last_policy"
        model_path = run_dir / model_name
        if (model_path.with_suffix(".zip")).exists():
            print(f"Loading RL weights from {model_path}...")
            agent.load(str(model_path))
        else:
            print(f"Warning: Weights {model_name} not found in {run_dir}. Training from current state.")
    from stable_baselines3.common.callbacks import CallbackList
    from src.rl.callbacks import CnnLoggingCallback
    
    eval_callback = NBVCallback(eval_env, logger, eval_freq=config.EVAL_FREQ, n_eval_episodes=config.N_EVAL_EPISODES, save_path=str(run_dir))
    cnn_callback = CnnLoggingCallback(run_dir)
    callback = CallbackList([eval_callback, cnn_callback])
    
    try:
        print(f"Starting training run: {run_id}")
        agent.train(total_timesteps=config.TOTAL_TIMESTEPS, callback=callback)
    except KeyboardInterrupt:
        print("Training interrupted manually.")
    finally:
        print("Saving last policy...")
        agent.save(str(run_dir / "last_policy"))
        
        # Plotting learning curves
        import matplotlib.pyplot as plt
        import json
        
        # Loading logs from Logger object directly if possible, or from file
        logs = logger.logs
        
        if not logs and os.path.exists(logger.jsonl_path):
            with open(logger.jsonl_path, 'r') as f:
                logs = [json.loads(line) for line in f if line.strip()]
                
        if logs:
            steps = [l["step"] for l in logs]
            rewards = [l["reward"] for l in logs]
            objs = [l["obj_metric"] for l in logs]
            reward_stds = [l["reward_std"] for l in logs]
            obj_stds = [l["obj_metric_std"] for l in logs]
            
            # Rewards plot
            plt.figure(figsize=(10, 5))
            plt.fill_between(steps, np.array(rewards)-np.array(reward_stds), np.array(rewards)+np.array(reward_stds), alpha=0.2)
            plt.plot(steps, rewards, label="Mean Reward (SAC, BasicNet)")
            plt.title("Reward Function")
            plt.xlabel("Timesteps")
            plt.ylabel("Reward")
            plt.legend()
            plt.savefig(run_dir / "reward_curve.png")
            plt.close()
            
            # Obj metric plot
            plt.figure(figsize=(10, 5))
            plt.fill_between(steps, np.array(objs)-np.array(obj_stds), np.array(objs)+np.array(obj_stds), alpha=0.2)
            plt.plot(steps, objs, label="Mean Accuracy Difference (SAC, BasicNet)", color='orange')
            plt.title("Objective Metric")
            plt.xlabel("Timesteps")
            plt.ylabel("Accuracy Diff")
            plt.legend()
            plt.savefig(run_dir / "objective_curve.png")
            plt.close()
            
            print("Plots saved.")
        else:
            print("Logs are empty or not found, skipping plot generation.")

if __name__ == "__main__":
    train()
