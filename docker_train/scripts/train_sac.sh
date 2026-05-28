#!/bin/bash
# train_sac.sh — SAC training entrypoint inside the container.
# All parameters come from environment variables (set via .env / docker-compose).
set -euo pipefail

# ── Signal forwarding ─────────────────────────────────────────
# Lets docker stop / Ctrl-C cleanly save a checkpoint.
_term() {
    echo "[train_sac] SIGTERM received — forwarding to Python PID $PY_PID"
    kill -TERM "$PY_PID" 2>/dev/null
}
trap _term SIGTERM SIGINT

# ── Parameters (env vars with sensible defaults) ──────────────
ODIN_WEIGHTS="${ODIN_WEIGHTS:-/data/weights/model_final.pth}"
ODIN_CFG="${ODIN_CFG:-/workspace/odin/configs/scannet_context/3d.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output}"
POLICY="${POLICY:-sac}"
TOTAL_STEPS="${TOTAL_STEPS:-200000}"
SCENE_STAGE="${SCENE_STAGE:-2}"
MAX_STEPS="${MAX_STEPS:-10}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-true}"
SWAG="${SWAG:-false}"
LEARNING_STARTS="${LEARNING_STARTS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
GRADIENT_STEPS="${GRADIENT_STEPS:-4}"
LR_ACTOR="${LR_ACTOR:-0.0001}"
LR_CRITIC="${LR_CRITIC:-0.0003}"

# ── Build optional flags ──────────────────────────────────────
EXTRA_FLAGS=""
[ "$FREEZE_BACKBONE" = "true" ] && EXTRA_FLAGS="$EXTRA_FLAGS --freeze_backbone"
[ "$SWAG"            = "true" ] && EXTRA_FLAGS="$EXTRA_FLAGS --swag"

mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo "  NBV SAC Training"
echo "  policy=$POLICY  steps=$TOTAL_STEPS  stage=$SCENE_STAGE"
echo "  weights=$ODIN_WEIGHTS"
echo "  cfg=$ODIN_CFG"
echo "  output=$OUTPUT_DIR"
echo "  freeze_backbone=$FREEZE_BACKBONE  swag=$SWAG"
echo "================================================================"

python /workspace/nbv_project/train_odin_sac_rl.py \
    --odin_weights   "$ODIN_WEIGHTS" \
    --odin_cfg       "$ODIN_CFG" \
    --output_dir     "$OUTPUT_DIR" \
    --policy         "$POLICY" \
    --total_steps    "$TOTAL_STEPS" \
    --scene_stage    "$SCENE_STAGE" \
    --max_steps      "$MAX_STEPS" \
    --learning_starts "$LEARNING_STARTS" \
    --batch_size     "$BATCH_SIZE" \
    --gradient_steps "$GRADIENT_STEPS" \
    --lr_actor       "$LR_ACTOR" \
    --lr_critic      "$LR_CRITIC" \
    $EXTRA_FLAGS \
    2>&1 | tee "$OUTPUT_DIR/train_sac.log" &

PY_PID=$!
wait "$PY_PID"
EXIT_CODE=$?

echo "[train_sac] Finished — exit code $EXIT_CODE"
exit $EXIT_CODE
