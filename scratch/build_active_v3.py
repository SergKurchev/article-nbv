import json
import os

with open("notebooks/temp_pull/strawpick-segpoinnet-my-odin-nbv-2-active.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 0 (markdown): description
nb['cells'][0]['source'] = [
    "# Обучение ODIN на NBV Stage3 Dataset (Multi-object + Obstacles, 3 Textures)\n",
    "Этот ноутбук обучает ODIN на датасете NBV Stage 3 с препятствиями (8 геометрических форм × 3 текстуры = 24 класса + препятствия)."
]

# Cell 1 (markdown): dataset info
nb['cells'][1]['source'] = [
    "## Информация о датасете\n",
    "\n",
    "**NBV Stage 3 Dataset**: https://www.kaggle.com/datasets/sergkurchevusa/nbv-stage-3-dataset-multi-object-obstacles-correct\n",
    "\n",
    "### Характеристики:\n",
    "- **Objects**: 2-10 примитивов (8 классов) + препятствия\n",
    "- **Textures**: 3 типа (red, mixed, green)\n",
    "- **Total Classes**: 24 (8 shapes × 3 textures)\n",
    "- **Images**: 224×224 RGB + depth + masks\n",
    "- **Cameras**: **5 кадров** на сцену"
]

# Cell 2 (code): config
nb['cells'][2]['source'] = [
    "import os\n",
    "import subprocess\n",
    "import sys\n",
    "import urllib.request\n",
    "\n",
    "# Базовая конфигурация проекта\n",
    "CONFIG = {\n",
    "    \"ODIN_DIR\": \"my_odin\",\n",
    "    \"ODIN_REPO_URL\": \"https://github.com/SergKurchev/my_odin.git\",\n",
    "    \"RL_DIR\": \"article-nbv\",\n",
    "    \"RL_REPO_URL\": \"https://github.com/SergKurchev/article-nbv.git\",\n",
    "    # Публичные веса из katefgroup/odin\n",
    "    \"ODIN_WEIGHTS_URL\": \"https://huggingface.co/katefgroup/odin/resolve/main/scannet_resnet_47.8_73.3_32k_1.5k.pth\",\n",
    "    \"ODIN_WEIGHTS_PATH\": \"my_odin/models/odin_scannet_context.pth\",\n",
    "    \"M2F_WEIGHTS_URL\": \"https://huggingface.co/katefgroup/odin/resolve/main/m2f_coco.pkl\",\n",
    "    \"M2F_WEIGHTS_PATH\": \"my_odin/models/model_final_5c90d4.pkl\",\n",
    "}\n"
]

# Cell 4 (code): installation (add article-nbv clone)
# Let's read Cell 4 source code and insert article-nbv clone
c4_src = "".join(nb['cells'][4]['source'])
target_to_insert = '    if not os.path.exists(CONFIG["ODIN_DIR"]):\n        print("\\n0. Cloning ODIN...")\n        subprocess.run(["git", "clone", "-q", "-b", "feature/nbv_dataset_process", CONFIG["ODIN_REPO_URL"], CONFIG["ODIN_DIR"]], check=True)'

replacement_code = target_to_insert + '\n\n    # 0b. Клонирование репозитория article-nbv для визуализаций\n    if not os.path.exists(CONFIG["RL_DIR"]):\n        print("\\n0b. Cloning article-nbv...")\n        subprocess.run(["git", "clone", "-q", CONFIG["RL_REPO_URL"], CONFIG["RL_DIR"]], check=True)'

c4_src = c4_src.replace(target_to_insert, replacement_code)
nb['cells'][4]['source'] = [line + "\n" for line in c4_src.split("\n")[:-1]] + [c4_src.split("\n")[-1]]

# Cell 5 (markdown): Stage 3
nb['cells'][5]['source'] = [
    "## 2. Диагностика датасета и создание splits\n",
    "\n",
    "NBV Stage 3 датасет имеет следующую структуру (проверено локально):"
]

# Cell 6 (code): diagnostics and splits
nb['cells'][6]['source'] = [
    "# Диагностика NBV Stage 3 датасета и создание splits\n",
    "import os\n",
    "import json\n",
    "from pathlib import Path\n",
    "\n",
    "# Автоматический поиск датасета\n",
    "DATASET_DIR = None\n",
    "possible_paths = [\n",
    "    \"/kaggle/input/datasets/sergkurchevusa/nbv-stage-3-dataset-multi-object-obstacles-correct/stage3\",\n",
    "    \"/kaggle/input/nbv-stage-3-dataset-multi-object-obstacles-correct/stage3\",\n",
    "    \"/kaggle/input/nbv-stage-3-dataset-multi-object-obstacles-correct\",\n",
    "    \"/kaggle/input/datasets/sergkurchevusa/nbv-stage-3-dataset-multi-object-obstacles-correct\",\n",
    "    \"./dataset/primitives/stage3\"\n",
    "]\n",
    "for p in possible_paths:\n",
    "    if os.path.exists(p):\n",
    "        try:\n",
    "            if any(e.startswith(\"sample_\") for e in os.listdir(p)):\n",
    "                DATASET_DIR = p\n",
    "                print(f\"✓ Found dataset directory: {p}\")\n",
    "                break\n",
    "        except:\n",
    "            pass\n",
    "\n",
    "if DATASET_DIR is None:\n",
    "    if os.path.exists(\"/kaggle/input\"):\n",
    "        for root, dirs, files in os.walk(\"/kaggle/input\"):\n",
    "            if \"stage3\" in root or \"stage_3\" in root:\n",
    "                if any(d.startswith(\"sample_\") for d in dirs):\n",
    "                    DATASET_DIR = root\n",
    "                    print(f\"✓ Autodetected dataset directory: {DATASET_DIR}\")\n",
    "                    break\n",
    "\n",
    "if DATASET_DIR is None:\n",
    "    DATASET_DIR = \"/kaggle/input/datasets/sergkurchevusa/nbv-stage-3-dataset-multi-object-obstacles-correct/stage3\"\n",
    "\n",
    "SPLITS_FILE = \"./nbv_stage3_splits.json\"\n",
    "\n",
    "print(\"=== Диагностика NBV Stage 3 датасета ===\")\n",
    "print(f\"DATASET_DIR: {DATASET_DIR}\")\n",
    "print(f\"DATASET_DIR exists: {os.path.exists(DATASET_DIR)}\")\n",
    "\n",
    "if os.path.exists(DATASET_DIR):\n",
    "    entries = sorted([e for e in os.listdir(DATASET_DIR) if e.startswith(\"sample_\")])\n",
    "    print(f\"\\nНайдено {len(entries)} сэмплов\")\n",
    "    print(f\"Первые 10: {entries[:10]}\")\n",
    "    print(f\"Последние 10: {entries[-10:]}\")\n",
    "\n",
    "    import random\n",
    "    random.seed(42)\n",
    "\n",
    "    n_samples = len(entries)\n",
    "    n_train = int(0.8 * n_samples)\n",
    "    n_val = int(0.1 * n_samples)\n",
    "\n",
    "    shuffled_entries = entries.copy()\n",
    "    random.shuffle(shuffled_entries)\n",
    "\n",
    "    splits = {\n",
    "        \"train\": shuffled_entries[:n_train],\n",
    "        \"val\": shuffled_entries[n_train:n_train + n_val],\n",
    "        \"test\": shuffled_entries[n_train + n_val:]\n",
    "    }\n",
    "\n",
    "    with open(SPLITS_FILE, 'w') as f:\n",
    "        json.dump(splits, f, indent=2)\n",
    "\n",
    "    print(f\"\\n=== Созданы splits (random_state=42) ===\")\n",
    "    for split_name, ids in splits.items():\n",
    "        print(f\"{split_name}: {len(ids)} сэмплов\")\n",
    "\n",
    "    # Проверим структуру одного сэмпла\n",
    "    sample_dir = Path(DATASET_DIR) / entries[0]\n",
    "    if sample_dir.exists():\n",
    "        print(f\"\\n=== Структура {entries[0]} ===\")\n",
    "        for item in sorted(sample_dir.iterdir()):\n",
    "            if item.is_dir():\n",
    "                count = len(list(item.iterdir()))\n",
    "                print(f\"  {item.name}/ ({count} файлов)\")\n",
    "            else:\n",
    "                print(f\"  {item.name}\")\n"
]

# Cell 7 (markdown): SWAG info
nb['cells'][7]['source'] = [
    "## 3. Запуск обучения с SWAG\n",
    "\n",
    "### Особенности датасета NBV Stage3 (vs Stage 2):\n",
    "- **24 класса + препятствия**: 8 примитивов × 3 текстуры\n",
    "- **Наличие препятствий**: Stage 3 сложнее, так как присутствуют перекрытия\n",
    "- **5 кадров** на сэмпл\n",
    "\n",
    "> **Важно**: `--num_frames 5` — это максимум. Можно уменьшить для экономии VRAM."
]

# Cell 8 (code): sync previous checkpoints
nb['cells'][8]['source'] = [
    "import shutil\n",
    "\n",
    "def sync_previous_output():\n",
    "    \"\"\"Копирует старые чекпоинты для --resume\"\"\"\n",
    "    input_base = \"/kaggle/input\"\n",
    "    target_output = \"./output_nbv_stage3_active\"\n",
    "    \n",
    "    if not os.path.exists(input_base):\n",
    "        return\n",
    "\n",
    "    for root, dirs, files in os.walk(input_base):\n",
    "        if (\"output_nbv_stage3_active\" in root or \"output\" in root) and any(f.endswith(\".pth\") for f in files):\n",
    "            print(f\"Нашел старые чекпоинты в: {root}\")\n",
    "            if not os.path.exists(target_output):\n",
    "                os.makedirs(target_output)\n",
    "            \n            for f in files:\n",
    "                src = os.path.join(root, f)\n",
    "                dst = os.path.join(target_output, f)\n",
    "                if not os.path.exists(dst):\n",
    "                    shutil.copy2(src, dst)\n",
    "            print(f\"Успешно скопировал {len(files)} файлов в {target_output}\")\n",
    "            return\n",
    "\n",
    "sync_previous_output()\n"
]

# Cell 11 (code): train command
nb['cells'][11]['source'] = [
    "# Используем DATASET_DIR и SPLITS_FILE для Stage 3\n",
    "CONFIG_FILE = \"my_odin/configs/scannet_context/3d.yaml\"\n",
    "\n",
    "# Ищем начальные веса (по приоритету):\n",
    "# 1. Чекпоинт текущего запуска Stage 3 (для --resume)\n",
    "# 2. Обученные веса Stage 2 active\n",
    "# 3. Базовые веса ScanNet\n",
    "INITIAL_WEIGHTS = CONFIG[\"ODIN_WEIGHTS_PATH\"]\n",
    "\n",
    "possible_weights = [\n",
    "    \"./output_nbv_stage3_active/model_final.pth\",\n",
    "    \"./output_nbv_stage3_active/model_best.pth\",\n",
    "    \"/kaggle/input/strawpick-segpoinnet-my-odin-nbv-2-active/output_nbv_stage2_active/model_final.pth\",\n",
    "    \"/kaggle/input/strawpick-segpoinnet-my-odin-nbv-2-active/model_final.pth\",\n",
    "    \"/kaggle/input/strawpick-segpoinnet-my-odin-nbv-2-active/output_nbv_stage2_active/model_best.pth\",\n",
    "]\n",
    "\n",
    "for w in possible_weights:\n",
    "    if os.path.exists(w):\n",
    "        INITIAL_WEIGHTS = w\n",
    "        print(f\"✓ Found initial weights: {w}\")\n",
    "        break\n",
    "\n",
    "train_cmd = [\n",
    "    VENV_PYTHON, \"my_odin/my_train_odin.py\",\n",
    "    \"--config-file\", CONFIG_FILE,\n",
    "    \"--num-gpus\", \"1\",\n",
    "    \"--dist-url\", \"tcp://127.0.0.1:23456\",\n",
    "    \"--resume\",\n",
    "    \"--visualize\",\n",
    "    \"--dataset_dir\", DATASET_DIR,\n",
    "    \"--splits_file\", SPLITS_FILE,\n",
    "    \n",
    "    # ПАРАМЕТРЫ ОБУЧЕНИЯ\n",
    "    \"--num_epochs\", \"50\",\n",
    "    \"--eval_period\", \"200\",\n",
    "    \"--checkpoint_period\", \"200\",\n",
    "    \n",
    "    # ОГРАНИЧЕНИЕ ВРЕМЕНИ\n",
    "    \"--max_time\", \"6\",\n",
    "    \n",
    "    # ПАРАМЕТРЫ РЕСУРСОВ\n",
    "    \"--image_size\", \"224\",\n",
    "    \"--batch_size\", \"3\",\n",
    "    \"--num_frames\", \"5\",\n",
    "    \"--lr\", \"0.00005\",\n",
    "    \"--warmup_ratio\", \"0.3\",\n",
    "    \n",
    "    # ТЕХНИЧЕСКИЕ ПАРАМЕТРЫ\n",
    "    'MODEL.WEIGHTS', INITIAL_WEIGHTS,\n",
    "    'OUTPUT_DIR', './output_nbv_stage3_active',\n",
    "    'SOLVER.AMP.ENABLED', 'True',\n",
    "    \n",
    "    # SWAG ПАРАМЕТРЫ\n",
    "    'MODEL.NBV_ACTIVE', 'True',\n",
    "    'MODEL.COVERAGE_HEAD.LOSS_WEIGHT', '1.0',\n",
    "    'MODEL.NBV_HEAD.LOSS_WEIGHT', '0.5',\n",
    "    'MODEL.BAYESIAN_TYPE', 'swag',\n",
    "    'MODEL.BAYESIAN_SAMPLES', '1',\n",
    "    'MODEL.BAYESIAN_INFERENCE_DURING_TRAINING', 'False',\n",
    "    'MODEL.SWAG.UPDATE_FREQ', '5',\n",
    "    'MODEL.SWAG.MAX_MODELS', '10',\n",
    "    'MODEL.SWAG.RANK', '20',\n",
    "    'MODEL.SWAG.NO_COV_MAT', 'False',\n",
    "]\n",
    "\n",
    "print(\"Starting SWAG training on NBV Stage3 dataset...\")\n",
    "print(f\"Initial weights: {INITIAL_WEIGHTS}\")\n",
    "print(\"Dataset: Multi-object (2-10) + Obstacles with 3 textures\")\n",
    "\n",
    "venv_env = make_venv_env()\n",
    "run_in_venv(train_cmd, env=venv_env)\n"
]

# Cell 13 (code): inference
nb['cells'][13]['source'] = [
    "# Ищем финальную обученную модель для inference\n",
    "FINAL_MODEL = './output_nbv_stage3_active/model_final.pth'\n",
    "if not os.path.exists(FINAL_MODEL):\n",
    "    if os.path.exists('./output_nbv_stage3_active/model_best.pth'):\n",
    "        FINAL_MODEL = './output_nbv_stage3_active/model_best.pth'\n",
    "\n",
    "inference_cmd = [\n",
    "    VENV_PYTHON, \"my_odin/my_train_odin.py\",\n",
    "    \"--config-file\", CONFIG_FILE,\n",
    "    \"--num-gpus\", \"1\",\n",
    "    \"--eval-only\",\n",
    "    \"--dataset_dir\", DATASET_DIR,\n",
    "    \"--splits_file\", SPLITS_FILE,\n",
    "    \n",
    "    'MODEL.WEIGHTS', FINAL_MODEL,\n",
    "    'OUTPUT_DIR', './output_nbv_stage3_active',\n",
    "    \n",
    "    # SWAG INFERENCE\n",
    "    'MODEL.NBV_ACTIVE', 'True',\n",
    "    'MODEL.COVERAGE_HEAD.LOSS_WEIGHT', '1.0',\n",
    "    'MODEL.NBV_HEAD.LOSS_WEIGHT', '0.5',\n",
    "    'MODEL.BAYESIAN_TYPE', 'swag',\n",
    "    'MODEL.BAYESIAN_SAMPLES', '10',\n",
    "    'MODEL.SWAG.SCALE', '1.0',\n",
    "]\n",
    "\n",
    "print(f\"Running SWAG inference using weights: {FINAL_MODEL}...\")\n",
    "venv_env = make_venv_env()\n",
    "run_in_venv(inference_cmd, env=venv_env)\n"
]

# Cell 14 (markdown): results
nb['cells'][14]['source'] = [
    "## Результаты\n",
    "\n",
    "Метрики сохраняются в:\n",
    "- `output_nbv_stage3_active/metrics_comparison.csv`\n",
    "- `output_nbv_stage3_active/swag_state.pth` - SWAG статистика\n",
    "- `output_nbv_stage3_active/model_final.pth` - финальная модель\n"
]

# Cell 15 (code): visualizations!
nb['cells'][15]['cell_type'] = 'code'
nb['cells'][15]['source'] = [
    "# 5. Генерация 3D визуализаций (HTML point clouds)\n",
    "import subprocess\n",
    "\n",
    "VIS_OUT_DIR = \"./output_nbv_stage3_active/visualizations\"\n",
    "print(\"Генерирую 3D визуализации для первых 5 сэмплов Stage 3...\")\n",
    "\n",
    "vis_cmd = [\n",
    "    VENV_PYTHON, \"article-nbv/scripts/generate_all_visualizations.py\",\n",
    "    \"--stage\", \"3\",\n",
    "    \"--max-samples\", \"5\",\n",
    "    \"--dataset-dir\", DATASET_DIR,\n",
    "    \"--out-dir\", VIS_OUT_DIR,\n",
    "    \"--python\", VENV_PYTHON\n",
    "]\n",
    "\n",
    "subprocess.run(vis_cmd, check=True)\n",
    "print(f\"Визуализации успешно сохранены в: {VIS_OUT_DIR}\")\n"
]

# Cell 16 (code): package ZIP
nb['cells'][16]['cell_type'] = 'code'
nb['cells'][16]['source'] = [
    "# 6. Упаковка результатов в ZIP\n",
    "import shutil\n",
    "from pathlib import Path\n",
    "\n",
    "print(\"Упаковываю результаты обучения и визуализации...\")\n",
    "results_dir = Path(\"kaggle_results\")\n",
    "results_dir.mkdir(exist_ok=True)\n",
    "\n",
    "# Копируем чекпоинты и визуализации\n",
    "output_dir = Path(\"./output_nbv_stage3_active\")\n",
    "if output_dir.exists():\n",
    "    shutil.copytree(output_dir, results_dir / \"output_nbv_stage3_active\", dirs_exist_ok=True)\n",
    "\n",
    "# Создаем ZIP-архив\n",
    "shutil.make_archive(\"kaggle_results_stage3\", \"zip\", results_dir)\n",
    "print(\"Результаты успешно упакованы в: kaggle_results_stage3.zip\")\n"
]

# Ensure output notebook directory exists
os.makedirs("notebooks", exist_ok=True)
with open("notebooks/strawpick-segpoinnet-my-odin-nbv-3-active.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("✓ Successfully generated notebooks/strawpick-segpoinnet-my-odin-nbv-3-active.ipynb")
