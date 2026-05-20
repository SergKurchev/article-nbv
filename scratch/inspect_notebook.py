import json
import sys

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

with open("notebooks/temp_pull/strawpick-segpoinnet-my-odin-nbv-2-active.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Number of cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    print(f"\n--- Cell {i} ({cell['cell_type']}) ---")
    source = cell.get("source", [])
    if isinstance(source, list):
        source_str = "".join(source)
    else:
        source_str = source
    # Print only first line to keep it clean
    first_lines = source_str.split("\n")[:3]
    for line in first_lines:
        print(f"  {line}")
