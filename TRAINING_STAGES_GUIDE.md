# NBV Training Stages - Полное руководство

## Обзор

Проект использует **прогрессивное обучение** через 3 стадии возрастающей сложности. Каждая стадия имеет свой датасет, обученную vision модель и RL агента.

---

## Stage 1: Single Object (Baseline)

### Описание

Самая простая стадия для установления baseline производительности.

**Характеристики:**
- 🎯 **Объектов**: 1 (на фиксированной позиции)
- 🚫 **Препятствий**: 0
- 📍 **Позиция объекта**: `[0.5, 0.0, z]` где `z` рассчитывается от высоты объекта
- 🎨 **Текстуры**: 3 типа (red, mixed gradient, green)
- 📊 **Классов**: 8 (примитивные формы)

**Цель обучения:**
- Научить агента находить оптимальный ракурс для одного объекта
- Установить baseline метрики
- Проверить работу vision модели

### Конфигурация

```python
# config.py
SCENE_STAGE = 1
NUM_CLASSES = 8
OBJECT_MODE = "primitives"
```

### Полный Pipeline

#### Шаг 1: Генерация примитивов

**Что делает:** Создаёт 8 примитивных 3D форм с правильными нормалями.

```bash
uv run python src/data/generate_primitives.py
```

**Выход:**
```
src/data/objects/primitives/
├── Object_01/  # Sphere
│   ├── Object_01.STL      # ASCII STL для визуализации
│   ├── mesh.json          # JSON формат для PyBullet
│   └── texture.png        # Текстура
├── Object_02/  # Cube
├── Object_03/  # Elongated Box
├── Object_04/  # Pyramid
├── Object_05/  # Cone
├── Object_06/  # Cylinder
├── Object_07/  # Hourglass
└── Object_08/  # Octahedron
```

**Детали реализации:**
- Использует процедурную генерацию mesh
- Counter-clockwise winding order для правильных нормалей
- Фильтрация вырожденных треугольников
- Автоматическое вычисление нормалей через cross product

#### Шаг 2: Генерация датасета

**Что делает:** Создаёт RGB-D датасет с множественными видами каждого объекта.

```bash
# Тестовый датасет (8 сэмплов, 1 на класс)
uv run python scripts/prepare_stage_datasets.py --stage 1 --samples 8

# Полный датасет (8000 сэмплов, 1000 на класс)
uv run python scripts/prepare_stage_datasets.py --stage 1 --samples 8000
```

**Параметры генерации:**
```python
DATASET_SAMPLES_PER_CLASS = 1000  # Сэмплов на класс
DATASET_VIEWS_PER_SAMPLE = 5      # Видов на сэмпл
DATASET_CAMERA_RADIUS_MIN = 0.3   # Мин. расстояние камеры
DATASET_CAMERA_RADIUS_MAX = 0.6   # Макс. расстояние камеры
```

**Процесс генерации:**
1. Загружает PyBullet симуляцию (headless)
2. Для каждого класса (0-7):
   - Загружает объект с текстурой (red/mixed/green по `class_id % 3`)
   - Размещает на фиксированной позиции
   - Генерирует препятствия (0 для Stage 1)
   - Проверяет коллизии
   - Для каждого вида:
     - Вычисляет позицию камеры (сферические координаты)
     - Захватывает RGB, depth, segmentation
     - Проверяет видимость объекта (мин. 50 пикселей)
     - Сохраняет данные
3. Сохраняет metadata (cameras.json, color_map.json)

**Выход:**
```
dataset/primitives/stage1/
├── sample_00000/
│   ├── rgb/
│   │   ├── 00000.png  # RGB изображение (224×224)
│   │   ├── 00001.png
│   │   └── ...
│   ├── depth/
│   │   ├── 00000.npy  # Depth карта (float32, метры)
│   │   ├── 00001.npy
│   │   └── ...
│   ├── masks/
│   │   ├── 00000.png  # Instance segmentation (цветная)
│   │   ├── 00001.png
│   │   └── ...
│   ├── labels/
│   │   ├── 00000.txt  # YOLO bounding boxes
│   │   ├── 00001.txt
│   │   └── ...
│   ├── cameras.json   # Camera metadata
│   └── color_map.json # Instance → Category mapping
├── sample_00001/
└── ...
```

**Верификация:**
- Автоматически проверяет целостность данных
- Проверяет наличие всех файлов
- Валидирует размеры изображений
- Проверяет metadata

#### Шаг 3: Обучение Vision модели

**Что делает:** Обучает CNN (MultiModalNet или LightweightODIN) на RGB-D данных.

```bash
# Настройка в config.py
CNN_ARCHITECTURE = "LightweightODIN"  # или "MultiModalNet"
CNN_BATCH_SIZE = 16
CNN_EPOCHS = 10
CNN_LR = 1e-3

# Обучение
uv run python src/vision/train_cnn.py
```

**Процесс обучения:**
1. Загружает датасет из `dataset/primitives/stage1/`
2. Создаёт DataLoader (80% train, 20% val)
3. Инициализирует модель:
   - **LightweightODIN**: Attention-based, 1.4M параметров
   - **MultiModalNet**: CNN+MLP fusion, ~800K параметров
4. Обучает с CrossEntropyLoss
5. Сохраняет метрики (loss, accuracy, F1, acc_diff)
6. Генерирует learning curves
7. Сохраняет веса:
   - `weights/primitives/multimodal_best.pt` (лучшая val loss)
   - `weights/primitives/multimodal_last.pt` (последняя эпоха)

**Выход:**
```
runs/cnn_train_v1.0_primitives_8obj_YYYYMMDD_HHMMSS/
├── cnn_training_log.csv       # Метрики по батчам
├── cnn_learning_curve.png     # Графики обучения
├── multimodal_best.pt         # Лучшие веса
└── multimodal_last.pt         # Последние веса
```

**Метрики:**
- Loss (CrossEntropy)
- Accuracy (top-1)
- F1 Score (macro)
- Accuracy Difference (для RL reward)

#### Шаг 4: Тест обучения

**Что делает:** Проверяет, что модель может обучаться на минимальном датасете.

```bash
uv run python scripts/test_training_single_sample.py --stage 1
```

**Процесс:**
1. Загружает 8 сэмплов (по 1 на класс)
2. Создаёт LightweightODIN
3. Тестирует forward pass
4. Обучает 2 эпохи
5. Проверяет, что loss уменьшается

**Ожидаемый результат:**
```
Model: LightweightODIN
Parameters: 1,416,641

--- Testing Forward Pass ---
Input shape: torch.Size([4, 4, 224, 224])
Output shape: torch.Size([4, 8])
✓ Success

--- Testing Training Step ---
Epoch 1: Loss=2.38, Accuracy=12.50%
Epoch 2: Loss=2.25, Accuracy=25.00%
✓ Training test passed!
```

#### Шаг 5: Обучение RL агента

**Что делает:** Обучает SAC агента находить оптимальные ракурсы.

```bash
# Headless (быстро)
uv run python train.py

# С GUI (медленно, для отладки)
uv run python train.py --gui
```

**Параметры:**
```python
TOTAL_TIMESTEPS = 500000
BATCH_SIZE = 128
LEARNING_RATE = 1e-2
MAX_STEPS_PER_EPISODE = 10
REWARD_SCALE = 10.0
PENALTY_COLLISION = -15.0
```

**Процесс обучения:**
1. Инициализирует NBV environment
2. Загружает обученную vision модель
3. Создаёт SAC агента (MultiInputPolicy)
4. Обучает с reward:
   ```python
   reward = accuracy_difference * REWARD_SCALE
   reward += PENALTY_COLLISION if collision else 0
   reward += PENALTY_OOB if out_of_bounds else 0
   ```
5. Периодически оценивает (каждые 2000 шагов)
6. Сохраняет веса и метрики

**Выход:**
```
runs/rl_train_v1.0_primitives_8obj_YYYYMMDD_HHMMSS/
├── best_policy.zip            # Лучшая policy
├── last_policy.zip            # Последняя policy
├── logs.jsonl                 # Метрики обучения
├── eval_rewards.png           # График наград
└── eval_accuracy_diff.png     # График accuracy difference
```

#### Шаг 6: Оценка агента

**Что делает:** Оценивает обученного агента на тестовых эпизодах.

```bash
uv run python evaluate.py
```

**Процесс:**
1. Загружает лучшую policy
2. Запускает N эпизодов (default: 10)
3. Собирает метрики:
   - Средняя награда
   - Средний accuracy difference
   - Количество коллизий
   - Количество OOB
4. Выводит статистику

#### Шаг 7: 3D Визуализация

**Что делает:** Создаёт интерактивный HTML viewer для анализа датасета.

```bash
# Для одного сэмпла
uv run python scripts/generate_sample_viewer.py dataset/primitives/stage1/sample_00000

# Для первых 3 сэмплов
uv run python scripts/generate_all_visualizations.py --stage 1 --max-samples 3
```

**Процесс:**
1. Читает все виды сэмпла (RGB, depth, masks)
2. Для каждого вида:
   - Unproject depth → 3D точки (учёт OpenGL конвенции)
   - Трансформирует в world coordinates
   - Добавляет цвет и segmentation labels
3. Объединяет все виды в единое облако точек
4. Генерирует HTML с Three.js
5. Добавляет camera frustums для визуализации позиций камер

**Выход:**
```
dataset/primitives/stage1/sample_00000/visualization.html
```

**Управление:**
- Scroll: zoom
- Left drag: orbit
- Right drag: pan
- Кнопки: RGB / Category / Instances

---

## Stage 2: Multiple Objects

### Описание

Средняя стадия с множественными объектами для обучения мультиобъектной классификации.

**Характеристики:**
- 🎯 **Объектов**: 2-10 (случайное размещение)
- 🚫 **Препятствий**: 0
- 📍 **Размещение**: Uniform distribution в bounds
- 🔍 **Коллизии**: Проверка между всеми объектами
- 📏 **Мин. расстояние**: `radius_1 + radius_2 + 0.05m`

**Цель обучения:**
- Научить агента работать с несколькими объектами
- Агрегировать uncertainty по всем объектам
- Находить ракурсы, оптимальные для всей сцены

### Конфигурация

```python
# config.py
SCENE_STAGE = 2
MIN_OBJECTS = 2
MAX_OBJECTS = 10

# Spatial bounds
SCENE_BOUNDS_X_MIN = 0.2
SCENE_BOUNDS_X_MAX = 0.8
SCENE_BOUNDS_Y_MIN = -0.3
SCENE_BOUNDS_Y_MAX = 0.3
SCENE_BOUNDS_Z_MIN = 0.15
SCENE_BOUNDS_Z_MAX = 0.4

# Collision detection
SCENE_MIN_OBJECT_DISTANCE = 0.25
SCENE_MAX_PLACEMENT_ATTEMPTS = 100
```

### Полный Pipeline

#### Шаг 1: Генерация датасета

```bash
# Тестовый
uv run python scripts/prepare_stage_datasets.py --stage 2 --samples 8

# Полный
uv run python scripts/prepare_stage_datasets.py --stage 2 --samples 8000
```

**Отличия от Stage 1:**
1. **Размещение объектов:**
   ```python
   # Для каждого объекта:
   for attempt in range(MAX_PLACEMENT_ATTEMPTS):
       # Случайная позиция в bounds
       pos = [
           random.uniform(X_MIN, X_MAX),
           random.uniform(Y_MIN, Y_MAX),
           random.uniform(Z_MIN, Z_MAX)
       ]
       
       # Проверка коллизий со всеми существующими объектами
       collision = False
       for other_obj in placed_objects:
           distance = np.linalg.norm(pos - other_obj.pos)
           min_dist = obj.radius + other_obj.radius + 0.05
           if distance < min_dist:
               collision = True
               break
       
       if not collision:
           place_object(pos)
           break
   ```

2. **Количество объектов:**
   - Случайное от MIN_OBJECTS до MAX_OBJECTS
   - Разные классы для разнообразия

3. **Segmentation:**
   - Каждый объект получает уникальный instance_id
   - Color map содержит все объекты сцены

#### Шаг 2: Обучение Vision модели

```bash
# Настройка
CNN_ARCHITECTURE = "LightweightODIN"

# Обучение
uv run python src/vision/train_cnn.py
```

**Отличия:**
- Модель учится классифицировать несколько объектов одновременно
- Accuracy вычисляется по всем объектам в сцене
- Более сложные сцены → может потребоваться больше эпох

#### Шаг 3: Обучение RL агента

```bash
uv run python train.py
```

**Reward function:**
```python
# Агрегация по всем объектам
acc_diffs = []
for obj in scene_objects:
    pred = model.predict(image, vector)
    acc_diff = pred[obj.class_id] - max(pred[other_classes])
    acc_diffs.append(acc_diff)

# Средний accuracy difference
reward = np.mean(acc_diffs) * REWARD_SCALE
reward += PENALTY_COLLISION if collision else 0
reward += PENALTY_OOB if out_of_bounds else 0
```

**Цель:**
- Найти ракурс, оптимальный для классификации ВСЕХ объектов
- Балансировать между видимостью разных объектов

#### Шаг 4: Оценка и визуализация

```bash
# Оценка
uv run python evaluate.py

# Визуализация
uv run python scripts/generate_all_visualizations.py --stage 2 --max-samples 3
```

---

## Stage 3: Multiple Objects + Obstacles

### Описание

Самая сложная стадия с препятствиями для обучения работе с окклюзиями.

**Характеристики:**
- 🎯 **Объектов**: 2-10 (случайное размещение)
- 🚧 **Препятствий**: 1-5 (серые панели)
- 📍 **Размещение**: Uniform distribution с проверкой коллизий
- 🔍 **Коллизии**: Объект↔Объект, Объект↔Препятствие, Препятствие↔Препятствие
- 🛡️ **Безопасность**: 0.25м от робота

**Цель обучения:**
- Научить агента обходить препятствия
- Находить ракурсы с минимальными окклюзиями
- Active search вокруг препятствий

### Конфигурация

```python
# config.py
SCENE_STAGE = 3
MIN_OBJECTS = 2
MAX_OBJECTS = 10
MIN_OBSTACLES = 1
MAX_OBSTACLES = 5

# Obstacle dimensions (random per obstacle)
# half_extents = [
#     random.uniform(0.01, 0.05),  # Thickness
#     random.uniform(0.1, 0.2),    # Width
#     random.uniform(0.1, 0.3)     # Height
# ]
```

### Полный Pipeline

#### Шаг 1: Генерация датасета

```bash
# Тестовый
uv run python scripts/prepare_stage_datasets.py --stage 3 --samples 8

# Полный
uv run python scripts/prepare_stage_datasets.py --stage 3 --samples 8000
```

**Процесс генерации:**
1. **Размещение объектов** (как в Stage 2)
2. **Генерация препятствий:**
   ```python
   for i in range(random.randint(MIN_OBSTACLES, MAX_OBSTACLES)):
       # Случайные размеры
       half_extents = [
           random.uniform(0.01, 0.05),  # Толщина
           random.uniform(0.1, 0.2),    # Ширина
           random.uniform(0.1, 0.3)     # Высота
       ]
       
       # Случайная позиция
       for attempt in range(MAX_PLACEMENT_ATTEMPTS):
           pos = random_position_in_bounds()
           
           # Проверка коллизий
           collision = check_collision_with_all(pos, half_extents)
           
           if not collision:
               create_obstacle(pos, half_extents, color=[128, 128, 128])
               break
   ```

3. **Проверка видимости:**
   - Объекты могут быть частично окклюдированы
   - Вид считается валидным, если видно хотя бы 50 пикселей объекта

4. **Segmentation:**
   - Target objects: красный
   - Robot: зелёный
   - Obstacles: синий (с градацией для разных препятствий)

#### Шаг 2: Обучение Vision модели

```bash
uv run python src/vision/train_cnn.py
```

**Особенности:**
- Модель учится работать с частичными окклюзиями
- Может потребоваться больше данных (больше видов на сэмпл)
- Accuracy может быть ниже из-за окклюзий

#### Шаг 3: Обучение RL агента

```bash
uv run python train.py
```

**Reward function:**
```python
# Та же агрегация, но с учётом окклюзий
reward = np.mean(acc_diffs) * REWARD_SCALE
reward += PENALTY_COLLISION if collision else 0  # -15.0 (строже!)
reward += PENALTY_OOB if out_of_bounds else 0
```

**Особенности обучения:**
- Агент учится избегать препятствий
- Находит ракурсы с минимальными окклюзиями
- Может потребоваться больше timesteps

#### Шаг 4: Оценка и визуализация

```bash
# Оценка
uv run python evaluate.py

# Визуализация
uv run python scripts/generate_all_visualizations.py --stage 3 --max-samples 3
```

**Метрики:**
- Occlusion handling rate
- Obstacle avoidance rate
- View diversity
- Average accuracy difference

---

## Мастер-скрипт для всех стадий

### Полный автоматический pipeline

```bash
# Тестовый запуск (8 сэмплов на стадию)
uv run python scripts/master_pipeline.py --samples 8 --test-only

# Полный pipeline (8000 сэмплов на стадию)
uv run python scripts/master_pipeline.py --samples 8000

# Только загрузка на Kaggle
uv run python scripts/master_pipeline.py --upload-only
```

**Что делает `master_pipeline.py`:**

1. **Генерация датасетов** (для каждой стадии):
   ```python
   for stage in [1, 2, 3]:
       run("scripts/prepare_stage_datasets.py", 
           f"--stage {stage} --samples {args.samples}")
   ```

2. **Тест обучения** (если не `--skip-test`):
   ```python
   for stage in [1, 2, 3]:
       run("scripts/test_training_single_sample.py",
           f"--stage {stage}")
   ```

3. **Загрузка на Kaggle** (если не `--test-only`):
   ```python
   for stage in [1, 2, 3]:
       run("scripts/upload_to_kaggle.py",
           f"--stage {stage} --username {user} --key {key}")
   ```

4. **Вывод сводки:**
   - Количество созданных сэмплов
   - Ссылки на Kaggle датасеты
   - Статус каждой стадии

---

## Сравнение стадий

| Параметр | Stage 1 | Stage 2 | Stage 3 |
|----------|---------|---------|---------|
| **Объектов** | 1 | 2-10 | 2-10 |
| **Препятствий** | 0 | 0 | 1-5 |
| **Размещение** | Фиксированное | Случайное | Случайное |
| **Коллизии** | Нет | Объект↔Объект | Все↔Все |
| **Сложность** | Низкая | Средняя | Высокая |
| **CNN Accuracy** | >90% | >85% | >80% |
| **RL Reward** | >5.0 | >3.0 | >2.0 |
| **Steps to converge** | <5 | <7 | <10 |

---

## Troubleshooting

### Проблема: Объекты касаются земли

**Решение:** Проверьте `SCENE_BOUNDS_Z_MIN` и safety margin:
```python
z_position = SCENE_BOUNDS_Z_MIN + half_height + 0.05  # 5см запас
```

### Проблема: Слишком много неудачных размещений

**Решение:** Увеличьте bounds или уменьшите количество объектов:
```python
SCENE_BOUNDS_X_MAX = 1.0  # Было 0.8
MAX_OBJECTS = 8  # Было 10
```

### Проблема: CNN не сходится

**Решение:** Уменьшите learning rate или увеличьте batch size:
```python
CNN_LR = 1e-4  # Было 1e-3
CNN_BATCH_SIZE = 32  # Было 16
```

### Проблема: RL агент сталкивается с препятствиями

**Решение:** Увеличьте collision penalty:
```python
PENALTY_COLLISION = -20.0  # Было -15.0
```

---

## Ожидаемое время выполнения (CPU)

### Тестовый pipeline (8 сэмплов на стадию)
- Генерация датасетов: ~5 минут на стадию
- Тест обучения: ~1 минута на стадию
- **Общее время**: ~20 минут

### Полный pipeline (8000 сэмплов на стадию)
- Генерация датасетов: ~2-3 часа на стадию
- CNN обучение: ~10-30 минут на стадию
- RL обучение: ~2-3 часа на стадию
- **Общее время**: ~15-20 часов

---

## Kaggle интеграция

### Загрузка датасетов

```bash
# Все стадии
uv run python scripts/upload_to_kaggle.py

# Конкретная стадия
uv run python scripts/upload_to_kaggle.py --stage 1

# Обновление существующего
uv run python scripts/upload_to_kaggle.py --update
```

### Скачивание датасетов

```bash
# Установка Kaggle CLI
pip install kaggle

# Скачивание
kaggle datasets download -d sergeykurchev/nbv-stage1-dataset
kaggle datasets download -d sergeykurchev/nbv-stage2-dataset
kaggle datasets download -d sergeykurchev/nbv-stage3-dataset

# Распаковка
unzip nbv-stage1-dataset.zip -d dataset/primitives/stage1/
unzip nbv-stage2-dataset.zip -d dataset/primitives/stage2/
unzip nbv-stage3-dataset.zip -d dataset/primitives/stage3/
```

---

## Заключение

Прогрессивное обучение через 3 стадии позволяет:
- ✅ Постепенно увеличивать сложность
- ✅ Отлаживать на простых сценариях
- ✅ Сравнивать производительность между стадиями
- ✅ Transfer learning между стадиями

**Рекомендуемый порядок:**
1. Начните с Stage 1 для baseline
2. Убедитесь, что CNN и RL работают
3. Переходите к Stage 2 для мультиобъектной классификации
4. Завершите Stage 3 для работы с окклюзиями

---

**Версия**: 1.0  
**Дата**: 2026-04-22  
**Автор**: NBV Project Team
