import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path

class NBVCallback(BaseCallback):
    def __init__(self, eval_env, logger, eval_freq=2000, n_eval_episodes=10, save_path=None, verbose=1):
        super(NBVCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.logger_custom = logger
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.best_mean_obj = -np.inf
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            rewards = []
            obj_metrics = []
            
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                episode_reward = 0.0
                episode_obj = []
                while not done:
                    action, _states = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    episode_obj.append(info.get("acc_diff", 0.0))
                    
                rewards.append(episode_reward)
                if episode_obj:
                    obj_metrics.append(np.max(episode_obj))
                else:
                    obj_metrics.append(0.0)
                
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            mean_obj = np.mean(obj_metrics)
            std_obj = np.std(obj_metrics)
            
            # Use total timesteps correctly handled by SB3
            total_steps = self.model.num_timesteps
            self.logger_custom.log_episode(total_steps, mean_reward, mean_obj, std_reward, std_obj)
            
            if self.verbose:
                print(f"Eval step {total_steps}: reward={mean_reward:.2f}, obj={mean_obj:.2f}")
                
            if mean_obj > self.best_mean_obj:
                self.best_mean_obj = mean_obj
                if self.save_path is not None:
                    # Save best policy
                    self.model.save(os.path.join(self.save_path, "best_policy"))
                    if self.verbose:
                        print("Saved new best model.")
                        
        return True
