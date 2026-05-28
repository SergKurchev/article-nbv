#!/bin/bash
# evaluate.sh — Policy comparison entrypoint inside the container.
# Runs evaluate.py for each policy in POLICIES and writes aggregate metrics.
set -euo pipefail

_term() {
    echo "[evaluate] SIGTERM received — forwarding to Python PID $PY_PID"
    kill -TERM "$PY_PID" 2>/dev/null
}
trap _term SIGTERM SIGINT

# ── Parameters ────────────────────────────────────────────────
ODIN_WEIGHTS="${ODIN_WEIGHTS:-/data/weights/model_final.pth}"
ODIN_CFG="${ODIN_CFG:-/workspace/odin/configs/scannet_context/3d.yaml}"
CHECKPOINT="${CHECKPOINT:-/workspace/output/best.pth}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-/workspace/eval_output}"
POLICIES="${POLICIES:-sac geometric uncertainty}"
NUM_EPISODES="${NUM_EPISODES:-50}"
SCENE_STAGE="${SCENE_STAGE:-2}"
SWAG="${SWAG:-false}"

EXTRA_FLAGS=""
[ "$SWAG" = "true" ] && EXTRA_FLAGS="$EXTRA_FLAGS --swag"

# If checkpoint doesn't exist, skip it and warn (baselines still run).
CKPT_FLAG=""
if [ -f "$CHECKPOINT" ]; then
    CKPT_FLAG="--checkpoint $CHECKPOINT"
else
    echo "[evaluate] WARNING: checkpoint not found at $CHECKPOINT — SAC policy uses random weights."
fi

mkdir -p "$EVAL_OUTPUT_DIR"

echo "================================================================"
echo "  NBV Policy Evaluation"
echo "  policies='$POLICIES'  episodes=$NUM_EPISODES  stage=$SCENE_STAGE"
echo "  checkpoint=$CHECKPOINT"
echo "  output=$EVAL_OUTPUT_DIR"
echo "================================================================"

python /workspace/nbv_project/evaluate.py \
    --odin_weights  "$ODIN_WEIGHTS" \
    --odin_cfg      "$ODIN_CFG" \
    --policies      $POLICIES \
    --num_episodes  "$NUM_EPISODES" \
    --scene_stage   "$SCENE_STAGE" \
    --output_dir    "$EVAL_OUTPUT_DIR" \
    $CKPT_FLAG \
    $EXTRA_FLAGS \
    2>&1 | tee "$EVAL_OUTPUT_DIR/evaluate.log" &

PY_PID=$!
wait "$PY_PID"
EXIT_CODE=$?

echo "[evaluate] Finished — exit code $EXIT_CODE"
exit $EXIT_CODE
