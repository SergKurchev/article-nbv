#!/bin/bash
# update_server.sh — Fast update without Docker rebuild.
# Pushes only: project Python source + scripts.
# Use this after editing .py files or shell scripts.
#
# Does NOT rebuild the image — the new Python files are picked up
# from the bind-mount at /workspace/scripts, and the project source
# is re-COPYd only on the next full build.
# For Python source changes: rebuild with   docker compose build
# For script-only changes:   just re-run this script, then restart container.
set -euo pipefail

SERVER="176.109.83.84"
PORT="2221"
USER="root"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="/root/skurchev/workspace/nbv_with_obstacles"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "--> Updating scripts and project source on server..."

rsync -av --progress \
    -e "ssh -p $PORT -i $SSH_KEY" \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    "$SCRIPT_DIR/scripts/" "$USER@$SERVER:$REMOTE_DIR/docker_train/scripts/"

rsync -av --progress \
    -e "ssh -p $PORT -i $SSH_KEY" \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.git/' \
    "$PROJECT_ROOT/src/"              "$USER@$SERVER:$REMOTE_DIR/src/"

for f in train_odin_sac_rl.py evaluate.py config.py; do
    rsync -av --progress \
        -e "ssh -p $PORT -i $SSH_KEY" \
        "$PROJECT_ROOT/$f" "$USER@$SERVER:$REMOTE_DIR/$f"
done

ssh -p "$PORT" -i "$SSH_KEY" "$USER@$SERVER" "
    chmod +x $REMOTE_DIR/docker_train/scripts/train_sac.sh
    chmod +x $REMOTE_DIR/docker_train/scripts/evaluate.sh
"

echo "--> Done. Restart container to pick up Python changes:"
echo "    ssh -p $PORT $USER@$SERVER"
echo "    cd $REMOTE_DIR/docker_train && docker compose restart nbv-train"
