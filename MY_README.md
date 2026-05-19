# NBV Project - Complete Guide

## Quick Start

### Generate Datasets (8 samples for testing)
```bash
uv run python scripts/prepare_stage_datasets.py --stage 1 --samples 8
uv run python scripts/prepare_stage_datasets.py --stage 2 --samples 8
uv run python scripts/prepare_stage_datasets.py --stage 3 --samples 8
```

### Generate Full Datasets (1000 samples per class)
```bash
uv run python scripts/prepare_stage_datasets.py --stage 1 --samples 800
uv run python scripts/prepare_stage_datasets.py --stage 2 --samples 800
uv run python scripts/prepare_stage_datasets.py --stage 3 --samples 800
```

### Generate 3D Visualizations
```bash
# First sample of each stage
uv run python scripts/generate_all_visualizations.py --max-samples 1

# Open in browser
start dataset/primitives/stage1/sample_00000/visualization.html
```

### Train Models
```bash
# CNN training
uv run python src/vision/train_cnn.py

# RL training (старый SAC-агент, без ODIN)
uv run python train.py

# Evaluation
uv run python evaluate.py
```

### Train ODIN NBV Head as RL Agent (REINFORCE)
```bash
# Frozen backbone (рекомендуется — быстрее, стабильнее)
uv run python train_odin_nbv_rl.py \
    --odin_weights /path/to/model_final.pth \
    --odin_cfg my_odin/configs/scannet_context/3d.yaml \
    --freeze_backbone \
    --total_episodes 500

# End-to-end (backbone тоже обучается — медленнее, больше GPU памяти)
uv run python train_odin_nbv_rl.py \
    --odin_weights /path/to/model_final.pth \
    --odin_cfg my_odin/configs/scannet_context/3d.yaml \
    --total_episodes 500

# Тесты (без GPU/ODIN зависимостей)
uv run python -m pytest tests/test_odin_nbv_agent.py -v
```

### Train ODIN NBV Head as RL Agent (SAC) — рекомендуется
```bash
# SAC: off-policy, replay buffer, twin Q-critics, auto entropy
uv run python train_odin_sac_rl.py \
    --odin_weights /path/to/model_final.pth \
    --odin_cfg my_odin/configs/scannet_context/3d.yaml \
    --freeze_backbone \
    --total_steps 100000

# End-to-end SAC
uv run python train_odin_sac_rl.py \
    --odin_weights /path/to/model_final.pth \
    --odin_cfg my_odin/configs/scannet_context/3d.yaml \
    --total_steps 100000

# Тесты SAC
uv run python -m pytest tests/test_odin_sac_agent.py -v
```

### Update Kaggle Datasets
```bash
# Check what will be uploaded
uv run python scripts/update_kaggle_datasets.py --dry-run

# Upload all stages
uv run python scripts/update_kaggle_datasets.py
```

---

## Training Stages

### Stage 1: Single Object (Baseline)
- **Objects**: 1 at fixed position `[0.5, 0.0, z]`
- **Obstacles**: 0
- **Purpose**: Baseline performance

### Stage 2: Multiple Objects
- **Objects**: 2-10 with uniform spatial distribution
- **Obstacles**: 0
- **Purpose**: Multi-object classification

### Stage 3: Multiple Objects + Obstacles
- **Objects**: 2-10 with uniform spatial distribution
- **Obstacles**: 1-5 (gray panels)
- **Purpose**: Occlusion handling

---

## ODIN NBV Head как RL-Агент (REINFORCE)

### Архитектура

```
PyBullet Simulator
       │
       ▼
   RGB-D кадр ──────────────────────────┐
       │                                │
       ▼                                ▼
  ODIN Backbone                   Coverage Head
  (ResNet + Pixel Decoder)        (scene features → p_hidden)
       │                                │
       ▼                                ▼
  NBV Head ◄── current_pose      Reward = Δp_hidden
       │
       ▼
  next_pos (3D) + next_quat (4D)  ←─  ACTION
       │
       ▼
  PyBullet: робот перемещает камеру
```

**Ключевая идея:** NBV Head — это **и есть политика** RL-агента.
Она принимает 3D-фичи сцены + текущую позу камеры и предсказывает
следующую лучшую позицию обзора. Алгоритм REINFORCE обновляет
веса NBV Head так, чтобы максимизировать снижение неопределённости
(p_hidden), которое оценивает Coverage Head.

**Нет отдельного SAC-агента.** Вся политика живёт внутри ODIN.

### Два режима заморозки

| Флаг | Что обучается | Что заморожено | GPU память | Скорость |
|------|---------------|----------------|------------|----------|
| `--freeze_backbone` | Только NBV Head | Backbone + Coverage Head | ~4 GB | Быстро |
| *(без флага)* | Вся сеть end-to-end | Ничего | ~12+ GB | Медленно |

- **`--freeze_backbone` (рекомендуется):** Backbone ODIN и Coverage Head
  заморожены. Градиенты REINFORCE обновляют **только** NBV Head.
  Coverage Head даёт стабильный сигнал награды (p_hidden не "плывёт").

- **Без флага (end-to-end):** Градиенты текут через весь backbone.
  Backbone адаптирует свои фичи под задачу NBV. Требует больше GPU
  и может быть нестабильным — используйте малый lr и grad_clip.

### Параметры CLI

```
--odin_weights PATH      Путь к .pth весам NBVActiveODIN
--odin_cfg PATH          Путь к YAML конфигу ODIN
--freeze_backbone        Заморозить backbone + Coverage Head
--total_episodes N       Число эпизодов (default: 500)
--lr FLOAT               Learning rate (default: 1e-4)
--exploration_std FLOAT  Начальный шум Gaussian (default: 0.1)
--std_decay FLOAT        Decay шума за эпизод (default: 0.999)
--std_min FLOAT          Минимальный шум (default: 0.01)
--gamma FLOAT            Discount factor (default: 0.99)
--grad_clip FLOAT        Max gradient norm (default: 1.0)
--scene_stage {1,2,3}    Стадия сцены (default: 2)
--save_freq N            Checkpoint каждые N эпизодов (default: 50)
--output_dir PATH        Папка для результатов
```

### Выходные файлы

```
output_nbv_rl/
├── nbv_head_best.pth        # Лучшая политика (NBV Head веса)
├── nbv_head_last.pth        # Последний checkpoint
├── nbv_head_ep50.pth        # Промежуточные checkpoints
├── nbv_head_ep100.pth
├── nbv_rl_metrics.csv       # Метрики по эпизодам
└── logs/
    └── training_metrics.csv # Пошаговые метрики среды
```

### SAC vs REINFORCE — какой алгоритм выбрать?

| | REINFORCE (`train_odin_nbv_rl.py`) | SAC (`train_odin_sac_rl.py`) |
|---|---|---|
| **Тип** | On-policy | Off-policy |
| **Replay Buffer** | Нет | Да (50k transitions) |
| **Sample efficiency** | Низкая | **Высокая** |
| **Стабильность** | Высокая дисперсия | Twin Q + entropy |
| **Скорость сходимости** | Медленная | **Быстрая** |
| **Рекомендуется для** | Отладка, малые эксперименты | **Полное обучение** |

### SAC: параметры CLI

```
--odin_weights PATH      Путь к .pth весам NBVActiveODIN
--odin_cfg PATH          Путь к YAML конфигу ODIN
--freeze_backbone        Заморозить backbone + Coverage Head
--total_steps N          Общее число шагов (default: 100000)
--lr_actor FLOAT         LR актора/NBV Head (default: 1e-4)
--lr_critic FLOAT        LR Q-networks (default: 3e-4)
--tau FLOAT              Soft update коэффициент (default: 0.005)
--buffer_size N          Размер replay buffer (default: 50000)
--batch_size N           Batch для SAC updates (default: 256)
--learning_starts N      Шагов до начала обучения (default: 1000)
--gamma FLOAT            Discount factor (default: 0.99)
--grad_clip FLOAT        Max gradient norm (default: 1.0)
--scene_stage {1,2,3}    Стадия сцены (default: 2)
--output_dir PATH        Папка для результатов
```

### SAC: выходные файлы

```
output_odin_sac/
├── best.pth             # Лучшая политика (NBV Head + Q-networks)
├── last.pth             # Последний checkpoint
├── ckpt_step5000.pth    # Промежуточные checkpoints
├── sac_metrics.csv      # Метрики по эпизодам
└── logs/
    └── training_metrics.csv
```

### Запуск на Kaggle

Датасет сцен **НЕ нужен** — среда PyBullet генерирует сцены динамически
из 3D-мешей, которые уже есть в репозитории. Нужны только:
1. **Веса ODIN** (`model_final.pth`) — загрузить как Kaggle Dataset.
2. **GPU** (T4 x2 рекомендуется).
3. **Internet: On** (для установки зависимостей).

Пример запуска SAC в Kaggle-ноутбуке:
```python
# SAC (рекомендуется)
!venv/bin/python nbv_rl/train_odin_sac_rl.py \
    --odin_weights /kaggle/input/nbv-odin-weights/model_final.pth \
    --odin_cfg my_odin/configs/scannet_context/3d.yaml \
    --freeze_backbone \
    --total_steps 50000 \
    --output_dir ./output_odin_sac

# REINFORCE (альтернатива)
!venv/bin/python nbv_rl/train_odin_nbv_rl.py \
    --odin_weights /kaggle/input/nbv-odin-weights/model_final.pth \
    --odin_cfg my_odin/configs/scannet_context/3d.yaml \
    --freeze_backbone \
    --total_episodes 300 \
    --output_dir ./output_nbv_rl
```

---

## Dataset Structure

Each sample contains:
```
sample_XXXXX/
├── rgb/                    # RGB images (224×224 PNG)
│   ├── 00000.png
│   └── ...
├── depth/                  # Depth maps (224×224 NPY, float32, meters)
│   ├── 00000.npy
│   └── ...
├── masks/                  # Instance segmentation (224×224 PNG, colored)
│   ├── 00000.png
│   └── ...
├── labels/                 # YOLO bounding boxes (TXT)
│   ├── 00000.txt
│   └── ...
├── cameras.json           # Camera metadata (position, target, rotation, intrinsics)
└── color_map.json         # Instance ID → Category mapping
```

---

## 3D Visualization

### Controls
- **Scroll**: Zoom in/out
- **Left drag**: Orbit around objects
- **Right drag / Middle**: Pan camera

### Display Modes
- **RGB**: Color as seen by camera
- **Category**: Color by category (target/robot/obstacles)
- **Instances**: Color by instance (each object different color)

### Additional Options
- **Cameras**: Show/hide camera frustums
- **Cam Dots**: Show/hide camera position dots
- **Reset View**: Return to initial camera position

---

## Kaggle Datasets

- **Stage 1**: https://www.kaggle.com/datasets/sergeykurchev/nbv-stage1-dataset
- **Stage 2**: https://www.kaggle.com/datasets/sergeykurchev/nbv-stage2-dataset
- **Stage 3**: https://www.kaggle.com/datasets/sergeykurchev/nbv-stage3-dataset

### Download from Kaggle
```bash
pip install kaggle

kaggle datasets download -d sergeykurchev/nbv-stage1-dataset
kaggle datasets download -d sergeykurchev/nbv-stage2-dataset
kaggle datasets download -d sergeykurchev/nbv-stage3-dataset

unzip nbv-stage1-dataset.zip -d dataset/primitives/stage1/
unzip nbv-stage2-dataset.zip -d dataset/primitives/stage2/
unzip nbv-stage3-dataset.zip -d dataset/primitives/stage3/
```

---

## Technical Details

### Camera Coordinates (OpenGL Convention)
- **Forward**: -Z
- **Up**: +Y
- **Right**: +X

### Depth Unprojection
```python
X_cam = (u - cx) * depth / fx
Y_cam = -(v - cy) * depth / fy  # Flip Y for OpenGL
Z_cam = -depth  # Forward is -Z
```

### Primitive Normals
All primitives use **counter-clockwise (CCW) winding order** when viewed from outside, ensuring normals point outward.

### Color Scheme (Category Mode)
- 🔴 **Red** - Target objects
- 🟢 **Green** - Robot (Kuka manipulator)
- 🔵 **Blue** - Obstacles (Stage 3 only)

---

## Key Scripts

### Dataset Generation
- `scripts/prepare_stage_datasets.py` - Generate and verify datasets
- `scripts/master_pipeline.py` - Full pipeline for all stages
- `src/data/generate_primitives.py` - Generate 8 primitive shapes

### Visualization
- `scripts/generate_sample_viewer.py` - Generate 3D visualization for one sample
- `scripts/generate_all_visualizations.py` - Batch generate visualizations

### Kaggle Upload
- `scripts/update_kaggle_datasets.py` - Automatic version updater

### Training
- `src/vision/train_cnn.py` - CNN training
- `train.py` - RL training (старый SAC, без ODIN)
- `train_odin_sac_rl.py` - **SAC: NBV Head = RL actor** ⭐ рекомендуется
- `train_odin_nbv_rl.py` - REINFORCE: NBV Head = RL policy
- `train_odin_rl.py` - SAC + ODIN как сенсор (устаревший)
- `evaluate.py` - Evaluation
- `gui.py` - Interactive GUI

### Tests
- `tests/test_odin_sac_agent.py` - Mock-тесты SAC + NBV Head (без GPU/ODIN)
- `tests/test_odin_nbv_agent.py` - Mock-тесты REINFORCE + NBV Head
- `tests/test_odin_adapter.py` - Тесты ODINAdapter

---

## Troubleshooting

### Visualization Issues
- **Objects not visible**: Press "Reset View" or try "Category" mode
- **Slow loading**: Reduce points with `--stride 4 --max-points 200000`
- **Browser compatibility**: Use Chrome 90+, Firefox 88+, or Edge 90+

### Dataset Generation
- **Objects touching ground**: Check `SCENE_BOUNDS_Z_MIN` and safety margin
- **Too many placement failures**: Increase bounds or reduce object count
- **CNN not converging**: Reduce learning rate or increase batch size

### Kaggle Upload
- **401 Unauthorized**: Check credentials in `~/.kaggle/kaggle.json`
- **Dataset not found**: Script will auto-create on first upload

---

## Changelog

### 2026-05-19 - RL Environment & Telemetry Enhancements
- ✅ **Исправление коллизий (`environment_odin.py`):** Проверка столкновений с препятствиями (obstacles) теперь выполняется всегда, независимо от режима работы манипулятора (`no_arm`). Коллизии с целевыми объектами проверяются только когда манипулятор активен.
- ✅ **Новые составляющие награды (Reward Shaping):**
  - Добавлена награда за **увеличение уверенности классификатора** (разница `mean_class_conf` между текущим и предыдущим шагом) со скейлингом **30.0**.
  - Добавлен разовый бонус **+20.0** за **фактическое нахождение всех объектов** на сцене (независимо от правильности их классификации).
  - Добавлен шаг выживания: **+1.0** за каждый шаг без столкновений и выхода за рабочую область.
- ✅ **Расширенная телеметрия (`training_metrics.csv`):** Добавлены подробные пошаговые метрики для отслеживания хода обучения агента:
  - `rew_coverage`, `rew_classifier`, `rew_all_found`, `rew_success_classified`, `rew_survival` — детальное логирование каждой составляющей общей награды на каждом шаге.
  - `objects_found_so_far` — количество найденных уникальных объектов с начала эпизода.
  - `all_objects_found_real` — найдены ли все объекты на сцене в действительности.
  - `model_confidence_all_found` — уверенность модели в том, что найдены все объекты (`1.0 - p_hidden`).
  - `classifier_confidence_mean` — средняя уверенность (score) классификационной головы ODIN по обнаруженным объектам.
- ✅ **Визуализация дашбордов и видео-телеметрия:**
  - Реализован двухпанельный видеоряд: первый показывает вид с камеры манипулятора, второй — общую 3D-сцену со стороны.
  - На текстовую панель добавлена детальная телеметрия положения камеры: 3D-координаты (`Pos`) и углы поворота камеры (`Rot`) в градусах (Roll, Pitch, Yaw).
  - Оптимизирован размер выходных видео-файлов: кадры записываются выборочно только для 1-го, 5-го, 10-го и завершающего шагов эпизода.
  - Исправлен баг «черного экрана» на Kaggle: добавлена константа `ER_TINY_RENDERER` в моки, а также динамический расчет окна скользящего среднего (окно = 5 при количестве эпизодов < 50), что позволяет отображать графики обучения с первых шагов.
- ✅ **Корректность тестов:** Все тесты в `tests/` и симуляции в `demo_video.py` успешно проходят.


### 2026-04-22 - Critical Fixes
- ✅ Fixed Stage 2: Now generates 2-10 objects WITHOUT obstacles
- ✅ Fixed Stage 3: Now generates 2-10 objects WITH obstacles (1-5)
- ✅ Fixed visualization: Objects no longer duplicate (uses correct camera target/up)
- ✅ Created stage-aware dataset generator (`src/vision/dataset_stage.py`)
- ✅ Added class tracking to `AssetLoader.generate_scene()`
- ✅ Removed all old incorrect datasets

---

**Version**: 1.1  
**Date**: 2026-05-19  
**Status**: Telemetry & Collision Fixes Ready ✅
