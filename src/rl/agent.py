from stable_baselines3 import SAC
import config
from pathlib import Path

class NBVAgent:
    def __init__(self, env):
        self.env = env
        self.model = SAC(
            "MultiInputPolicy",
            env,
            learning_rate=config.LEARNING_RATE,
            buffer_size=config.BUFFER_SIZE,
            learning_starts=config.LEARNING_STARTS,
            batch_size=config.BATCH_SIZE,
            tau=0.005,
            gamma=0.99,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1
        )
        
    def train(self, total_timesteps, callback=None):
        self.model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=4)
        
    def save(self, path):
        self.model.save(path)
        
    def load(self, path):
        self.model = SAC.load(path, env=self.env)
        
    def predict(self, obs, deterministic=True):
        return self.model.predict(obs, deterministic=deterministic)
