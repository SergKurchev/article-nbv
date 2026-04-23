import json
import os
import shutil
import csv
from pathlib import Path
import numpy as np

class Logger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.json_path = self.run_dir / "logs.json"
        self.jsonl_path = self.run_dir / "logs.jsonl"
        self.csv_path = self.run_dir / "rl_metrics.csv"
        self.logs = []

        # 1. Load legacy JSON if exists (convert to JSONL if needed)
        if self.json_path.exists():
            try:
                with open(self.json_path, 'r') as f:
                    self.logs = json.load(f)
                # Convert to JSONL once
                if not self.jsonl_path.exists():
                    self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.jsonl_path, 'w') as f:
                        for entry in self.logs:
                            f.write(json.dumps(entry) + '\n')
                print(f"Migrated legacy logs from {self.json_path.name}")
            except Exception as e:
                print(f"Warning: Could not load legacy logs: {e}")

        # 2. Load JSONL if exists
        elif self.jsonl_path.exists():
            try:
                with open(self.jsonl_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            self.logs.append(json.loads(line))
            except Exception as e:
                print(f"Warning: Could not load JSONL logs: {e}")

        # 3. Create CSV header if doesn't exist
        if not self.csv_path.exists():
            try:
                self.csv_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['step', 'reward', 'obj_metric', 'reward_std', 'obj_metric_std'])
                print(f"Created RL metrics CSV: {self.csv_path}")
            except Exception as e:
                print(f"Warning: Could not create CSV: {e}")
                
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

        # Append to JSONL immediately (append mode)
        try:
            with open(self.jsonl_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            print(f"Error writing to JSONL file: {e}")

        # Also write to CSV
        try:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step, reward, obj_metric, reward_std, obj_metric_std])
        except Exception as e:
            print(f"Error writing to CSV: {e}")
            
    def save(self):
        # Already handled by append in log_episode for JSONL
        pass

    def get_last_step(self):
        if not self.logs:
            return 0
        return self.logs[-1]["step"]
