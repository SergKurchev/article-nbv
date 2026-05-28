#!/bin/bash
# run_docker_test.sh — надёжный запуск тестового прогона через heredoc SSH
set -euo pipefail

SERVER="176.109.83.84"
PORT="2221"
REMOTE_USER="root"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="/root/skurchev/workspace/nbv_with_obstacles"

SSH="ssh -p $PORT -i $SSH_KEY -o StrictHostKeyChecking=no"

echo "==================================================================="
echo "  Запуск тестового прогона в Docker на сервере: $SERVER:$PORT"
echo "==================================================================="

# Загружаем скрипт запуска прямо на сервер, обходя все проблемы с кавычками
$SSH $REMOTE_USER@$SERVER 'bash -s' << 'REMOTE_SCRIPT'
set -e
REMOTE_DIR="/root/skurchev/workspace/nbv_with_obstacles"

echo "--> Удаляем старый мёртвый контейнер если есть..."
docker rm -f nbv-eval-test 2>/dev/null && echo "  Старый контейнер удалён" || echo "  Старых контейнеров нет"

echo "--> Создаём папку для результатов..."
mkdir -p "$REMOTE_DIR/test_run_output"

echo "--> Запускаем evaluate.py внутри контейнера с GPU..."
docker run --rm --gpus all \
    --name nbv-eval-test \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e PYBULLET_HEADLESS=1 \
    -e DISPLAY= \
    -v "$REMOTE_DIR/weights:/data/weights:ro" \
    -v "$REMOTE_DIR/test_run_output:/workspace/eval_output" \
    -v "$REMOTE_DIR/src:/workspace/nbv_project/src" \
    -v "$REMOTE_DIR/evaluate.py:/workspace/nbv_project/evaluate.py" \
    -v "$REMOTE_DIR/config.py:/workspace/nbv_project/config.py" \
    nbv-odin-train:latest \
    -c "python /workspace/nbv_project/evaluate.py \
        --odin_weights /data/weights/model_best.pth \
        --odin_cfg /workspace/odin/configs/scannet_context/3d.yaml \
        --policies sac \
        --num_episodes 3 \
        --vis_episodes 0 1 2 \
        --scene_stage 2 \
        --output_dir /workspace/eval_output" \
    2>&1 | tee "$REMOTE_DIR/test_run_output/test.log"

echo "==================================================================="
echo "  Тест завершён! Результаты: $REMOTE_DIR/test_run_output/"
echo "==================================================================="
REMOTE_SCRIPT
