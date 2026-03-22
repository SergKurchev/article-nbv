import json
import os
import shutil
from pathlib import Path
import numpy as np

class Logger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.json_path = self.run_dir / "logs.json"
        self.logs = []
        
        # Load existing if any
        if self.json_path.exists():
            with open(self.json_path, 'r') as f:
                self.logs = json.load(f)
                
    def copy_config(self):
        import config
        config_src = Path(config.__file__)
        config_dst = self.run_dir / "config_copy.py"
        shutil.copy(config_src, config_dst)
        
    def log_episode(self, step, reward, obj_metric, reward_std, obj_metric_std):
        entry = {
            "step": int(step),
            "reward": float(reward),
            "obj_metric": float(obj_metric),
            "reward_std": float(reward_std),
            "obj_metric_std": float(obj_metric_std)
        }
        self.logs.append(entry)
        self.save()
        
    def save(self):
        # We write atomically
        tmp_path = self.json_path.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(self.logs, f, indent=4)
        tmp_path.replace(self.json_path)

    def get_last_step(self):
        if not self.logs:
            return 0
        return self.logs[-1]["step"]
