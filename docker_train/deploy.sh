#!/bin/bash
# deploy.sh — First-time deploy to the training server.
# Syncs the full project + builds the Docker image remotely.
#
# Usage (from project root or docker_train/):
#   bash docker_train/deploy.sh
set -euo pipefail

SERVER="176.109.83.84"
PORT="2221"
USER="root"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="/root/skurchev/workspace/nbv_with_obstacles"

SSH="ssh -p $PORT -i $SSH_KEY"
SCP="scp -P $PORT -i $SSH_KEY"

# Resolve project root (works whether called from project root or docker_train/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "==================================================================="
echo "  Deploying NBV-with-obstacles to $SERVER:$PORT"
echo "  local : $PROJECT_ROOT"
echo "  remote: $REMOTE_DIR"
echo "==================================================================="

# ── Create remote directory structure ────────────────────────
$SSH "$USER@$SERVER" "
    mkdir -p $REMOTE_DIR/docker_train/scripts
    mkdir -p $REMOTE_DIR/weights
    mkdir -p $REMOTE_DIR/output
    mkdir -p $REMOTE_DIR/eval_output
"

# ── Sync project source ───────────────────────────────────────
echo "--> Syncing project source..."
rsync -av --progress \
    -e "ssh -p $PORT -i $SSH_KEY" \
    --exclude='.venv/' \
    --exclude='kaggle_output/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    --exclude='article/' \
    --exclude='output_odin_sac/' \
    --exclude='eval_results/' \
    --exclude='odin_weights/' \
    --exclude='dataset/' \
    "$PROJECT_ROOT/" "$USER@$SERVER:$REMOTE_DIR/"

# ── Copy ODIN weights ─────────────────────────────────────────
echo "--> Copying ODIN weights..."
rsync -av --progress \
    -e "ssh -p $PORT -i $SSH_KEY" \
    "$PROJECT_ROOT/odin_weights/model_best.pth" "$USER@$SERVER:$REMOTE_DIR/weights/model_best.pth"

# ── Set script permissions on server ─────────────────────────
echo "--> Setting script permissions..."
$SSH "$USER@$SERVER" "
    chmod +x $REMOTE_DIR/docker_train/scripts/train_sac.sh
    chmod +x $REMOTE_DIR/docker_train/scripts/evaluate.sh
    chmod +x $REMOTE_DIR/docker_train/deploy.sh
    chmod +x $REMOTE_DIR/docker_train/update_server.sh
    chmod +x $REMOTE_DIR/docker_train/run_docker_test.sh
    # Create .env from example if not already there
    [ -f $REMOTE_DIR/docker_train/.env ] || cp $REMOTE_DIR/docker_train/.env.example $REMOTE_DIR/docker_train/.env
"

# ── Build Docker image on server ─────────────────────────────
echo "--> Starting Docker build on server (screen session: nbv_build)..."
$SSH "$USER@$SERVER" "
    nohup bash -c '
        cd $REMOTE_DIR/docker_train && \
        docker compose build 2>&1 | tee build.log
        echo \"Exit: \$?\" >> build.log
    ' > /dev/null 2>&1 &
"

echo ""
echo "==================================================================="
echo "  Build started in screen session 'nbv_build'."
echo ""
echo "  Monitor build:"
echo "    ssh -p $PORT $USER@$SERVER \"screen -r nbv_build\""
echo "    ssh -p $PORT $USER@$SERVER \"tail -f $REMOTE_DIR/docker_train/build.log\""
echo ""
echo "  After build completes — start training:"
echo "    ssh -p $PORT $USER@$SERVER"
echo "    cd $REMOTE_DIR/docker_train && docker compose up -d nbv-train"
echo ""
echo "  Watch training logs:"
echo "    docker logs -f nbv-train"
echo "==================================================================="
