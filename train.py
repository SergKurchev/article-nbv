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
    run_id = str(uuid.uuid4())[:8]
    run_dir = config.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    video_dir = run_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Custom JSON Logger
    logger = Logger(run_dir)
    logger.copy_config()
    
    # Load CNN
    vision_model = NetLoader.load()
    # If weights exist for vision model, load them here
    # vision_model.load_state_dict(torch.load("..."))
    
    # Create env
    env = DummyVecEnv([make_env(vision_model, args.no_arm, args.headless)])
    eval_env = DummyVecEnv([make_env(vision_model, args.no_arm, True)])
    
    agent = NBVAgent(env)
    
    # Load if specified
    if args.load != "none":
        # Load logic if picking specific run dir (would need param or find latest)
        print(f"Loading {args.load} weights not fully implemented without specific run_id. Training from scratch.")
        pass

    callback = NBVCallback(eval_env, logger, eval_freq=config.EVAL_FREQ, n_eval_episodes=config.N_EVAL_EPISODES, save_path=str(run_dir))
    
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
        with open(logger.json_path, 'r') as f:
            logs = json.load(f)
            
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

if __name__ == "__main__":
    train()
