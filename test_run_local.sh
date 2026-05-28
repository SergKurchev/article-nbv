#!/bin/bash
# test_run_local.sh — Run a local test evaluation of the architecture.
# Uses the real ODIN weights, initializes SAC randomly,
# calculates metrics, and generates HTML visualizations of point clouds.

set -euo pipefail

# Ensure we are in the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ODIN_WEIGHTS="odin_weights/model_best.pth"
ODIN_CFG="scratch/my_odin/configs/scannet_context/3d.yaml"
OUTPUT_DIR="test_run_output"

if [ ! -f "$ODIN_WEIGHTS" ]; then
    echo "ERROR: ODIN weights not found at $ODIN_WEIGHTS"
    exit 1
fi

if [ ! -f "$ODIN_CFG" ]; then
    echo "ERROR: ODIN config not found at $ODIN_CFG"
    exit 1
fi

echo "================================================================"
echo "  Local Test Run: SAC (Random Weights) + Visualization"
echo "  Weights: $ODIN_WEIGHTS"
echo "  Config:  $ODIN_CFG"
echo "  Output:  $OUTPUT_DIR"
echo "================================================================"

# Run evaluate.py on the 'sac' policy for 3 episodes, saving HTML for all 3.
# We don't provide --checkpoint so it uses randomly initialized weights for SAC.
python evaluate.py \
    --odin_weights "$ODIN_WEIGHTS" \
    --odin_cfg "$ODIN_CFG" \
    --policies sac \
    --num_episodes 3 \
    --vis_episodes 0 1 2 \
    --scene_stage 2 \
    --output_dir "$OUTPUT_DIR"

echo "================================================================"
echo "  Test run finished."
echo "  Metrics and CSVs are saved in: $OUTPUT_DIR/"
echo "  HTML Point Cloud Visualizations are in: $OUTPUT_DIR/vis/"
echo "================================================================"
