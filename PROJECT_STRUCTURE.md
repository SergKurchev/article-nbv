# Структура проекта - Финальная версия

## 📚 Документация (5 файлов)

### Основные файлы:

1. **README.md** (17K)
   - Описание проекта NBV
   - Архитектуры моделей (MultiModalNet, LightweightODIN, SimpleNet, PointNet)
   - Reward function и MDP спецификация
   - Основные команды для работы

2. **DATASETS_AND_VISUALIZATION.md** (8.3K) ← НОВЫЙ
   - Описание 3 стадий датасетов
   - Генерация датасетов
   - 3D визуализация (управление, режимы)
   - Kaggle интеграция
   - Технические детали (координаты, нормали)

3. **PROJECT_PIPELINE.md** (14K)
   - Полный pipeline для всех 3 стадий
   - Команды для генерации, обучения, оценки
   - Debug команды
   - Troubleshooting

4. **POINTNET_SUMMARY.md** (7.4K)
   - PointNet архитектура
   - Генерация point cloud датасетов
   - Обучение PointNet

5. **POINTNET_GUIDE.md** (7.7K)
   - Детальное руководство по PointNet
   - Примеры использования

## 🗂️ Структура директорий

```
NBV_with_obstacles_and_robot/
├── README.md                           # Основная документация
├── DATASETS_AND_VISUALIZATION.md       # Датасеты и визуализация
├── PROJECT_PIPELINE.md                 # Pipeline для 3 стадий
├── POINTNET_SUMMARY.md                 # PointNet документация
├── POINTNET_GUIDE.md                   # PointNet руководство
├── config.py                           # Конфигурация
│
├── dataset/
│   └── primitives/
│       ├── stage1/                     # Stage 1: Single Object
│       │   ├── sample_00000/
│       │   │   ├── rgb/
│       │   │   ├── depth/
│       │   │   ├── masks/
│       │   │   ├── labels/
│       │   │   ├── cameras.json
│       │   │   └── color_map.json
│       │   └── ... (7 сэмплов)
│       ├── stage2/                     # Stage 2: Multi-Object
│       │   └── ... (8 сэмплов)
│       └── stage3/                     # Stage 3: Multi-Object + Obstacles
│           └── ... (8 сэмплов)
│
├── scripts/
│   ├── prepare_stage_datasets.py       # Генерация датасетов
│   ├── test_training_single_sample.py  # Тест обучения
│   ├── upload_to_kaggle.py             # Загрузка на Kaggle
│   ├── master_pipeline.py              # Мастер-скрипт
│   ├── generate_sample_viewer.py       # 3D визуализация
│   └── generate_all_visualizations.py  # Массовая генерация
│
├── src/
│   ├── data/
│   │   ├── generate_primitives.py      # Генерация 8 примитивов
│   │   └── objects/
│   │       └── primitives/
│   │           ├── Object_01/          # Sphere
│   │           ├── Object_02/          # Cube
│   │           ├── Object_03/          # Elongated Box
│   │           ├── Object_04/          # Pyramid
│   │           ├── Object_05/          # Cone
│   │           ├── Object_06/          # Cylinder
│   │           ├── Object_07/          # Hourglass
│   │           └── Object_08/          # Octahedron
│   │
│   ├── vision/
│   │   ├── models.py                   # MultiModalNet, LightweightODIN, PointNet
│   │   ├── dataset.py                  # RGB-D dataset generator
│   │   ├── dataset_pointnet.py         # Point cloud dataset
│   │   ├── train_cnn.py                # CNN training
│   │   ├── train_pointnet.py           # PointNet training
│   │   └── texture_generator.py        # Texture generation
│   │
│   ├── simulation/
│   │   ├── environment.py              # NBV Gym environment
│   │   ├── asset_loader.py             # Scene generation (3 stages)
│   │   ├── camera.py                   # Camera with intrinsics
│   │   └── robot.py                    # Kuka manipulator
│   │
│   └── rl/
│       ├── agent.py                    # SAC agent wrapper
│       └── callbacks.py                # Training callbacks
│
├── train.py                            # RL training
├── evaluate.py                         # Evaluation
└── gui.py                              # Interactive GUI
```

## ✅ Что было сделано

### 1. Созданы датасеты
- 3 стадии (Single Object, Multi-Object, Multi-Object + Obstacles)
- 8 сэмплов на стадию (1 на класс)
- Все данные верифицированы

### 2. Исправлены координаты
- Правильная трансформация PyBullet → World coordinates
- Учёт OpenGL конвенции
- Единое облако точек в визуализациях

### 3. Исправлены нормали
- Counter-clockwise winding order для всех примитивов
- Внешняя сторона объектов видна
- Перегенерированы все примитивы и датасеты

### 4. Создана инфраструктура
- Скрипты для генерации датасетов
- Скрипты для 3D визуализации
- Интеграция с Kaggle
- Полная документация

## 🚀 Быстрый старт

### Генерация датасетов
```bash
# Тест (8 сэмплов)
uv run python scripts/master_pipeline.py --samples 8 --test-only

# Полный (8000 сэмплов)
uv run python scripts/master_pipeline.py --samples 8000
```

### Визуализация
```bash
# Генерация для первых 3 сэмплов каждой стадии
uv run python scripts/generate_all_visualizations.py --max-samples 3

# Открытие
start dataset/primitives/stage1/sample_00000/visualization.html
```

### Обучение
```bash
# CNN
uv run python src/vision/train_cnn.py

# RL
uv run python train.py

# Оценка
uv run python evaluate.py
```

## 📊 Kaggle датасеты

- Stage 1: https://www.kaggle.com/datasets/sergeykurchev/nbv-stage1-dataset
- Stage 2: https://www.kaggle.com/datasets/sergeykurchev/nbv-stage2-dataset
- Stage 3: https://www.kaggle.com/datasets/sergeykurchev/nbv-stage3-dataset

---

**Версия**: 1.0 Final  
**Дата**: 2026-04-22  
**Статус**: Production Ready ✅
