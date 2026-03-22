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

| Событие / Условие | Награда |
| :--- | :--- |
| **Out-of-Bounds (OOB) or Collision** (Агент выбросил объект или столкнулся с препятствием) | $-10$ |
| **Valid Step** (Штатный шаг) | $+ (\text{Accuracy Difference} \times 10)$ |

*Note on Accuracy Difference:* If $p_{\text{target}}$ is the predicted probability of the true target class, and $p_{\text{max\_err}}$ is the highest probability assigned to an incorrect class:
$$\text{Accuracy Difference}_t = p_{\text{target}}^t - p_{\text{max\_err}}^t$$

## Visualizations and Architecture
The project utilizes a **MultiModalNet** (Hybrid CNN-MLP) featuring:
- **CNN Encoder**: Processes 4-channel RGB-D images.
- **MLP Encoder**: Processes the 15D kinematic vector.
- **Dual Heads**: 
  - **Classification**: 18-class object identification.
  - **Segmentation**: Pixel-wise object mask prediction.

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
- **Figure 4**: t-SNE / UMAP clustering of the penultimate layer activations taken during the highest-confidence step.

## Основные команды (Usage)

Для начала работы с проектом убедитесь, что у вас установлен пакетный менеджер `uv`. Установите все зависимости в виртуальное окружение командой:
```bash
uv sync
```

### Генерация датасета и предобучение CNN (Computer Vision)
Генерация изображений объектов со случайных ракурсов для предварительного обучения классификатора `BasicNet`:
```bash
uv run python src/vision/dataset.py
```
После того как датасет сгенерирован (создана папка `dataset/`), запустите обучение CNN. Веса будут сохранены в `weights/multimodal_best.pt`:
```bash
uv run python src/vision/train_cnn.py
```


### Обучение агента (RL SAC)
Запуск обучения агента. Вы можете передать флаги `--headless` для режима без отрисовки (максимальная скорость) и `--no_arm` для режима летающего объекта:
```bash
uv run python train.py --headless
```

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
- `s`: Сделать один шаг (агент принимает решение на основе текущего `obs`).
- `a`: Переключить автоматический режим (агент делает шаги непрерывно).
- `r`: Сбросить эпизод.
- **Обзор**: Стрелки, Shift, Ctrl, зажатая клавиша мыши и колесико используются для управления камерой наблюдателя.
