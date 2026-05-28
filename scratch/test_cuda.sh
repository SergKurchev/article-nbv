#!/bin/bash
SERVER="176.109.83.84"
PORT="2221"

# Создаём тестовый python-файл на сервере
ssh -p $PORT -o StrictHostKeyChecking=no root@$SERVER 'cat > /tmp/check_cuda.py << '"'"'PYEOF'"'"'
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("PyTorch version:", torch.__version__)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PYEOF'

# Запускаем внутри нашего образа
ssh -p $PORT -o StrictHostKeyChecking=no root@$SERVER \
    'docker run --rm --gpus all \
        -v /tmp/check_cuda.py:/check_cuda.py \
        nbv-odin-train:latest \
        -c "python /check_cuda.py"'
