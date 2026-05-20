import csv
from pathlib import Path

csv_path = Path("demo_output/training_metrics.csv")
if not csv_path.exists():
    print(f"Error: {csv_path} does not exist.")
    exit(1)

with open(csv_path, mode="r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print("Total rows:", len(rows))
headers = ["episode", "step", "p_hidden", "reward", "rew_coverage", "rew_classifier", "rew_all_found", "rew_survival", "objects_found_so_far"]
print(" | ".join(headers))
print("-" * 120)
for r in rows[-5:]:
    row_str = " | ".join(f"{r[h]}" for h in headers)
    print(row_str)
