# Next Best View (NBV) Reinforcement Learning

## Problem Definition
The Next Best View (NBV) problem involves finding the optimal camera viewpoint (or optimally orienting an object in front of a fixed camera) to maximize the confidence of a downstream computer vision algorithm. In this project, a Reinforcement Learning agent (Soft Actor-Critic) controls a 7-DOF Kinova manipulator to reposition a target object. The goal is to orient the object such that a pre-trained **MultiModalNet** Convolutional Neural Network can classify it with maximum certainty, overcoming spawned obstacles that actively occlude the view.

## Environment Specification

### Observation Space
The state $s_t \in \mathcal{S}$ is a multi-modal **Gymnasium Dict**:
1. `image`: [4, 224, 224] RGB-D image (RGB + Linearized Depth).
2. `vector`: 15-dimensional continuous feature vector:
   - $X, Y, Z, Q_x, Q_y, Q_z, Q_w$: Position and Orientation of end-effector.
   - $\text{Accuracy Difference}$: Visual confidence metric.
   - $J_1, \dots, J_7$: Angular positions of the 7 robot joints.

### Action Space
The action $a_t \in \mathcal{A}$ is a 6-dimensional continuous vector representing the **absolute target pose**:
$$a_t = [x, y, z, \text{roll}, \text{pitch}, \text{yaw}]$$
This pose is achieved via Inverse Kinematics (IK) to position the robot's end-effector.

### Transition Dynamics
The PyBullet engine steps the physics forward according to the IK controls. The state transitions to $s_{t+1}$ based on the physical limits, collisions, and visual properties recalculated by the simulated camera and vision module.

## Reward Function
The reward $r_t$ is defined sequentially to penalize catastrophic failures and reward increased visual information.

| Событие / Условие                                                                          | Награда                                    |
| :----------------------------------------------------------------------------------------- | :----------------------------------------- |
| **Out-of-Bounds (OOB)** (Объект слишком далеко или под столом)                             | $-10$                                      |
| **Collision** (Столкновение с препятствием или робота с объектом)                          | $-15$                                      |
| **Valid Step** (Штатный шаг)                                                               | $+ (\text{Accuracy Difference} \times 10)$ |

*Note on Accuracy Difference:* If $p_{\text{target}}$ is the predicted probability of the true target class, and $p_{\text{max\_err}}$ is the highest probability assigned to an incorrect class:
$$\text{Accuracy Difference}_t = p_{\text{target}}^t - p_{\text{max\_err}}^t$$

## Visualizations and Architecture
The project supports multiple neural network architectures for visual classification:

### Three-Class Texture System

The project uses a **3-class color system** to represent object ripeness:

1. **Class 0: Fully Red** (`red.png`)
   - RGB: (255, 0, 0)
   - Represents ripe objects (e.g., ripe strawberries)

2. **Class 1: Red-Green Gradient** (`mixed.png`)
   - Smooth curved gradient from red to green
   - Generated using random Bezier curves
   - Each color occupies 30-70% of texture area
   - Represents partially ripe objects

3. **Class 2: Fully Green** (`green.png`)
   - RGB: (0, 255, 0)
   - Represents unripe objects

**Texture Generation:**
```bash
# Generate all three texture variants
uv run python src/vision/texture_generator.py

# Test textures interactively on objects
uv run python scripts/test_textures.py
```

**Texture Mapping:**
- Objects are assigned textures based on `class_id % 3`
- Object_01, Object_04, Object_07 → red texture
- Object_02, Object_05, Object_08 → mixed texture
- Object_03, Object_06 → green texture

### Available Models

#### 1. MultiModalNet (Default)
Hybrid CNN-MLP architecture featuring:
- **CNN Encoder**: Processes 4-channel RGB-D images through hierarchical convolutions
- **MLP Encoder**: Processes 15D kinematic vector (end-effector pose + joints)
- **Fusion Layer**: Concatenates image and vector features
- **Classification Head**: 18-class object identification

#### 2. LightweightODIN
Simplified ODIN-inspired architecture optimized for CPU inference with Bayesian uncertainty estimation:
- **Hierarchical CNN Encoder**: RGB-D feature extraction (32→64→128→256 channels)
- **Pixel Decoder**: Lightweight replacement for MSDeformAttn (1×1 convolutions)
- **Query Embeddings**: Learnable object queries (similar to DETR/ODIN)
- **Self-Attention**: Queries attend to each other (4 heads, hidden_dim=128)
- **Cross-Attention**: Queries attend to image features
- **Vector Fusion**: Robot state integrated via broadcast addition
- **Classification Head**: Per-query binary classification
- **MC Dropout**: Bayesian uncertainty estimation via Monte Carlo Dropout

**Uncertainty Estimation:**
```python
# Standard inference
logits = model(image, vector)

# Inference with uncertainty (20 MC samples)
mean_probs, std_probs, entropy = model.predict_with_uncertainty(image, vector, mc_samples=20)
# mean_probs: [B, num_classes] - mean predicted probabilities
# std_probs: [B, num_classes] - epistemic uncertainty (model uncertainty)
# entropy: [B] - predictive entropy (total uncertainty)
```

**Key simplifications from full ODIN:**
- Single attention layer instead of 6-9 transformer layers
- Standard convolutions instead of MSDeformAttn
- No cross-view attention (single-view processing)
- No 3D backprojection
- Smaller feature dimensions (128 vs 256)

**Performance:** ~2-3x lighter than MultiModalNet, suitable for CPU inference while maintaining attention mechanisms and uncertainty quantification.

#### 3. SimpleNet
Minimal baseline for debugging:
- Single Conv2d layer + pooling + linear classifier
- No vector input support

### Model Selection
Models are loaded via `NetLoader.load(arch="ModelName")`:
```python
# In src/vision/train_cnn.py or config.py
model = NetLoader.load(arch="LightweightODIN", num_classes=8, vector_dim=15)
```

The Soft Actor-Critic (SAC) network uses the **MultiInputPolicy** to process both image and vector streams simultaneously.

## Mathematical Notation Reference
- $s_t$, $a_t$, $r_t$: Concrete representations of state, action, and reward at timestep $t$.
- $S_t$, $A_t$, $R_t$: Random variables characterizing the MDP tuple.
- $\gamma \in [0, 1)$: Discount factor.
- $\theta, \pi^\theta$: Policy network parameters and the policy itself.
- $v^\pi(s) = \mathbb{E}_{A \sim \pi^\theta}[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} | S_t = s]$: Value function evaluated under policy $\pi$.
- $\mathbb{E}_{A \sim \pi^\theta}[\dots]$: Mathematical expectation over the policy distribution.

## Chart & Figure Conventions
All artifacts produced during training and evaluation are annotated following standard scientific practices:
- **Figure 1**: Reward Function. Learning curve depicting the mean cumulative reward and standard deviation intervals across timesteps.
- **Figure 2**: Objective Metric. Curve plotting the Accuracy Difference per evaluation step.
- **Figure 3**: Normalized Confusion Matrix mapping predicted vs true class distributions.
- **Figure 4**: t-SNE / UMAP clustering of the penultimate layer activations.
- **Log Files**: Training metrics are saved in `runs/<run_name>/logs.jsonl` in a robust append-only format.

## Основные команды (Usage)

Для начала работы с проектом убедитесь, что у вас установлен пакетный менеджер `uv`. Установите все зависимости в виртуальное окружение командой:
```bash
uv sync
```

### Генерация датасета и предобучение CNN (Computer Vision)
Генерация датасета с правильной структурой для предварительного обучения классификатора `MultiModalNet`:
```bash
uv run python src/vision/dataset.py
```

**Структура датасета:**
Датасет генерируется в формате `sample_NNNNN/` с несколькими кадрами облёта вокруг каждого объекта:
```
dataset/primitives/
├── sample_00000/
│   ├── cameras.json      # Интринсики и экстринсики всех камер
│   ├── color_map.json    # Маппинг RGB цветов к instance_id
│   ├── rgb/              # RGB кадры (00000.png - 00019.png)
│   ├── depth/            # Карты глубины (00000.npy - 00019.npy)
│   ├── masks/            # RGB маски с цветами из color_map
│   └── labels/           # YOLO bounding box аннотации
├── sample_00001/
└── ...
```

**Параметры генерации (config.py):**
- `DATASET_SAMPLES_PER_CLASS`: Количество сэмплов на класс (по умолчанию 1000)
- `DATASET_VIEWS_PER_SAMPLE`: Количество кадров вокруг объекта (по умолчанию 20)
- `DATASET_MIN_VALID_VIEWS`: Минимум валидных кадров для сохранения сэмпла (10)
- `DATASET_MIN_OBJECT_PIXELS`: Минимум видимых пикселей объекта (50)
- `DATASET_CAMERA_RADIUS_MIN/MAX`: Диапазон расстояния камеры от объекта (0.3-0.6 м)
- `DATASET_CAMERA_PHI_MIN/MAX`: Диапазон угла возвышения камеры (0.1-1.37 рад)

После генерации датасета запустите обучение CNN. Веса будут сохранены в `weights/multimodal_best.pt`:
```bash
uv run python src/vision/train_cnn.py
```

**Выбор архитектуры:**
По умолчанию используется `MultiModalNet`. Для использования `LightweightODIN` измените в `config.py`:
```python
CNN_ARCHITECTURE = "LightweightODIN"  # Options: "MultiModalNet", "LightweightODIN", "SimpleNet"
```

**Сравнение моделей:**
- `MultiModalNet`: Стандартная CNN+MLP, быстрое обучение, хорошая baseline
- `LightweightODIN`: Attention-based, лучше для сложных сцен с окклюзиями, медленнее на CPU (~1.5-2x)
- `SimpleNet`: Минимальная модель для отладки

**Параметры обучения (config.py):**
- `CNN_ARCHITECTURE`: Выбор архитектуры модели
- `CNN_LR`: Learning rate (по умолчанию 1e-3)
- `CNN_BATCH_SIZE`: Размер батча (16)
- `CNN_EPOCHS`: Количество эпох (10)
- `CNN_VAL_SPLIT`: Доля валидационной выборки (0.2)


### Обучение агента (RL SAC)
Запуск обучения агента. По умолчанию работает в **headless** режиме (без GUI) для максимальной скорости.
```bash
# Обычный запуск (вслепую)
uv run python train.py

# Запуск с визуализацией (PyBullet GUI)
uv run python train.py --gui

# Продолжение обучения с последних весов (last_policy)
uv run python train.py --load last

# Продолжение обучения с лучших весов (best_policy)
uv run python train.py --load best
```

### Автоматизация Kaggle (Cloud Training)
Для быстрого обучения на GPU Kaggle используйте готовые скрипты:
1. **Загрузка датасета**: `python upload_to_kaggle.py` (автоматически запакует и подготовит метаданные).
2. **Обучение на Kaggle**: Используйте `kaggle_train_vision.py` как обертку в вашем Notebook.

### Оценка обученной модели (Evaluation)
Запуск инференса готовой модели (из последней сессии или с флагом `--load best`) для извлечения признаков и построения UMAP/t-SNE графов, а также матрицы ошибок:
```bash
uv run python evaluate.py
```

### Интерактивная симуляция (GUI)
Запуск ручного или автоматического режима для визуального контроля и графической отладки в PyBullet:
```bash
uv run python gui.py
```
**Управление:**
- `n`: Сделать один шаг (агент принимает решение на основе текущего `obs`).
- `m`: Переключить автоматический режим (агент делает шаги непрерывно).
- `r`: Сбросить эпизод.
- **Обзор**: Стрелки, Shift, Ctrl, зажатая клавиша мыши и колесико используются для управления камерой наблюдателя.
