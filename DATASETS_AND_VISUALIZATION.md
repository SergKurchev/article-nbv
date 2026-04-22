# NBV Dataset & Visualization Guide

## Датасеты для обучения

Проект использует 3 отдельных датасета для прогрессивного обучения:

### Stage 1: Single Object (Baseline)
- **Сцена**: 1 объект на фиксированной позиции
- **Препятствия**: 0
- **Назначение**: Baseline для начального обучения
- **Директория**: `dataset/primitives/stage1/`

### Stage 2: Multiple Objects
- **Сцена**: 2-10 объектов со случайным размещением
- **Препятствия**: 0
- **Назначение**: Обучение мультиобъектной классификации
- **Директория**: `dataset/primitives/stage2/`

### Stage 3: Multiple Objects + Obstacles
- **Сцена**: 2-10 объектов + 1-5 препятствий
- **Препятствия**: Серые панели (случайные размеры)
- **Назначение**: Обучение работе с окклюзиями
- **Директория**: `dataset/primitives/stage3/`

## Генерация датасетов

### Быстрая генерация (тест)
```bash
# Генерация 8 сэмплов (1 на класс) для всех стадий
uv run python scripts/master_pipeline.py --samples 8 --test-only
```

### Полная генерация
```bash
# Генерация 8000 сэмплов (1000 на класс) для всех стадий
uv run python scripts/master_pipeline.py --samples 8000
```

### Генерация отдельной стадии
```bash
# Stage 1
uv run python scripts/prepare_stage_datasets.py --stage 1 --samples 8000

# Stage 2
uv run python scripts/prepare_stage_datasets.py --stage 2 --samples 8000

# Stage 3
uv run python scripts/prepare_stage_datasets.py --stage 3 --samples 8000
```

## Структура датасета

Каждый сэмпл содержит:

```
sample_XXXXX/
├── rgb/                    # RGB изображения (224×224 PNG)
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
├── depth/                  # Depth карты (224×224 NPY, float32, метры)
│   ├── 00000.npy
│   ├── 00001.npy
│   └── ...
├── masks/                  # Instance segmentation (224×224 PNG, цветные)
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
├── labels/                 # YOLO bounding boxes (TXT)
│   ├── 00000.txt
│   ├── 00001.txt
│   └── ...
├── cameras.json           # Camera metadata (position, rotation, intrinsics)
└── color_map.json         # Instance ID → Category mapping
```

## 3D Визуализация

### Генерация визуализаций

```bash
# Для одного сэмпла
uv run python scripts/generate_sample_viewer.py dataset/primitives/stage1/sample_00000

# Для первых 3 сэмплов каждой стадии (рекомендуется)
uv run python scripts/generate_all_visualizations.py --max-samples 3

# С параметрами
uv run python scripts/generate_sample_viewer.py dataset/primitives/stage1/sample_00000 \
    --stride 2 \
    --max-points 500000
```

### Открытие визуализации

Просто откройте файл `visualization.html` в браузере:
- Двойной клик на файл
- Или перетащите в окно браузера
- Или: `start dataset/primitives/stage1/sample_00000/visualization.html`

### Управление

**Камера:**
- **Scroll** - Zoom (приближение/отдаление)
- **Left drag** - Orbit (вращение вокруг объекта)
- **Right drag / Middle** - Pan (перемещение)

**Режимы отображения:**
- **RGB** - Цветное изображение (как видит камера)
- **Category** - Цвет по категориям (target/robot/obstacles)
- **Instances** - Цвет по экземплярам (каждый объект свой цвет)

**Дополнительно:**
- **Cameras** - Показать/скрыть camera frustums
- **Cam Dots** - Показать/скрыть точки позиций камер
- **Reset View** - Вернуть камеру в исходное положение

## Kaggle датасеты

Все датасеты загружены на Kaggle НО они все всего лишь на 8 сэмплов. Необходимо их сделать на 1000 сэмплов каждый и обновить. :

- **Stage 1**: https://www.kaggle.com/datasets/sergeykurchev/nbv-stage1-dataset
- **Stage 2**: https://www.kaggle.com/datasets/sergeykurchev/nbv-stage2-dataset
- **Stage 3**: https://www.kaggle.com/datasets/sergeykurchev/nbv-stage3-dataset

### Скачивание с Kaggle

```bash
# Установка Kaggle CLI
pip install kaggle

# Настройка credentials (уже настроено)
# ~/.kaggle/kaggle.json

# Скачивание
kaggle datasets download -d sergeykurchev/nbv-stage1-dataset
kaggle datasets download -d sergeykurchev/nbv-stage2-dataset
kaggle datasets download -d sergeykurchev/nbv-stage3-dataset

# Распаковка
unzip nbv-stage1-dataset.zip -d dataset/primitives/stage1/
unzip nbv-stage2-dataset.zip -d dataset/primitives/stage2/
unzip nbv-stage3-dataset.zip -d dataset/primitives/stage3/
```

## Обучение на датасетах

### CNN обучение

```bash
# Stage 1
# Редактировать config.py: SCENE_STAGE = 1
uv run python src/vision/train_cnn.py

# Stage 2
# Редактировать config.py: SCENE_STAGE = 2
uv run python src/vision/train_cnn.py

# Stage 3
# Редактировать config.py: SCENE_STAGE = 3
uv run python src/vision/train_cnn.py
```

### RL обучение

```bash
# После обучения CNN
# Редактировать config.py: SCENE_STAGE = 1/2/3
uv run python train.py

# Оценка
uv run python evaluate.py

# Интерактивный GUI
uv run python gui.py
```

## Технические детали

### Координаты камеры

Проект использует PyBullet с OpenGL конвенцией:
- **Forward**: -Z
- **Up**: +Y
- **Right**: +X

Unprojection depth → 3D:
```python
X_cam = (u - cx) * depth / fx
Y_cam = -(v - cy) * depth / fy  # Flip Y для OpenGL
Z_cam = -depth  # Forward is -Z
```

### Нормали примитивов

Все примитивы используют **counter-clockwise (CCW) winding order** при взгляде снаружи, что обеспечивает правильное направление нормалей наружу.

### Цветовая схема (Category mode)

- 🔴 **Красный** - Target object (целевой объект)
- 🟢 **Зелёный** - Robot (манипулятор Kuka)
- 🔵 **Синий** - Obstacles (препятствия, только Stage 3)

## Скрипты

### Основные скрипты

- `scripts/prepare_stage_datasets.py` - Генерация и верификация датасетов
- `scripts/test_training_single_sample.py` - Тест обучения на минимальном датасете
- `scripts/upload_to_kaggle.py` - Загрузка датасетов на Kaggle
- `scripts/master_pipeline.py` - Мастер-скрипт для полного pipeline
- `scripts/generate_sample_viewer.py` - Генерация 3D визуализации
- `scripts/generate_all_visualizations.py` - Массовая генерация визуализаций

### Вспомогательные скрипты

- `src/data/generate_primitives.py` - Генерация 8 примитивных форм
- `src/vision/texture_generator.py` - Генерация текстур (red/mixed/green)

## Troubleshooting

### Визуализация не открывается
- Используйте современный браузер (Chrome 90+, Firefox 88+, Edge 90+)
- Попробуйте другой браузер

### Слишком медленно загружается
Уменьшите количество точек:
```bash
uv run python scripts/generate_sample_viewer.py <sample_path> \
    --stride 4 \
    --max-points 200000
```

### Не видно объектов
- Нажмите "Reset View"
- Попробуйте режим "Category"
- Проверьте, что depth карты не пустые

---

**Версия**: 1.0  
**Дата**: 2026-04-22
