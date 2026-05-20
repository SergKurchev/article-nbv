# NBV-with-ODIN: Полное описание архитектуры и вердикт по обучению

> **Назначение этого документа**: передать полную техническую картину проекта следующему агенту/чату.  
> Репозиторий: `SergKurchev/article-nbv`  
> Последний коммит содержит все описанные изменения. Тесты: **44 passed, 0 failed**.

---

## 1. Общая идея проекта

**Задача**: Next Best View (NBV) — автоматически выбирать следующую позицию камеры для максимального покрытия сцены с объектами при минимуме столкновений.

**Подход**: Использовать предобученную 3D-instance-segmentation модель **ODIN** как backbone для vision, а её NBV Head — как политику (актора) в SAC (Soft Actor-Critic).

---

## 2. Стек технологий

| Компонент | Технология |
|-----------|-----------|
| Симуляция | PyBullet |
| Vision backbone | ODIN (Detectron2-based 3D instance segmentation) |
| RL-алгоритм | SAC (Soft Actor-Critic, off-policy) |
| Фреймворк | PyTorch |
| RL-среда | Custom Gymnasium `NBVODINEnv` |
| Среда запуска | Kaggle (GPU) |

---

## 3. Детальная архитектура

### 3.1 Пайплайн целиком

```
PyBullet симуляция
    │
    ▼
RGB-D (224×224) + поза камеры
    │
    ▼
ODINAdapter._run_backprojection()
    │   RGB-D → 3D облако точек (multi-scale XYZ)
    ▼
NBVActiveODIN (ODIN backbone)
    │
    ├──► Coverage Head → p_hidden ∈ [0,1]   (сигнал для reward)
    │       sigmoid(logit) = вероятность что скрытые зоны ЕСТЬ
    │       Обучается: BCE vs GT (1.0 если скрытые объекты есть, 0.0 если нет)
    │
    ├──► NBV Head → nbv_pos (3,) + nbv_quat (4,)  (mean действия актора)
    │       Это и есть RL-ПОЛИТИКА (actor)
    │
    └──► query_features (B, Q, D=256)  → scene_embedding (256,)
             усредняется по Q (query tokens трансформера)
             добавляется к obs_vector для Q-critic
    │
    ▼
OdinSACAgent
    │
    ├──► Actor: NBVHead(nbv_pos, nbv_quat) + learnable log_std(6,)
    │       action = Normal(mean, exp(log_std)).rsample()   ← reparameterization trick
    │
    ├──► Twin Q-Critic: QNetwork(obs_dim=274, action_dim=6)
    │
    └──► Entropy: auto-tuning α, target_entropy = -6
```

---

### 3.2 Пространство наблюдений (`NBVODINEnv`)

**Два потока:**

**`obs["image"]`** — `(4, 224, 224)` float32 ∈ [0,1]
- Каналы 0-2: RGB нормализованный
- Канал 3: Depth нормализованный

**`obs["vector"]`** — `(18,)` float32:

| Индексы | Содержимое | Размер |
|---------|-----------|--------|
| `[0:3]` | Позиция камеры (x, y, z) | 3 |
| `[3:7]` | Ориентация камеры (quaternion xyzw) | 4 |
| `[7]` | `delta_p_hidden` = p_hidden_prev - p_hidden_cur | 1 |
| `[8:15]` | Состояние суставов манипулятора (или нули если `no_arm=True`) | 7 |
| `[15:18]` | NBV hint = `nbv_pos` из последнего ODIN inference | 3 |

**Для Q-critic obs расширяется до (274,)**:  
`obs_vec_full = concat([obs["vector"] (18), scene_embedding (256)])`

---

### 3.3 Пространство действий

**`action`** — `(6,)` float32:
- `action[:3]` = целевая позиция камеры (x, y, z)
- `action[3:6]` = целевая ориентация (roll, pitch, yaw)

**Bounds** (из `config.py`):
```
ACTION_MIN = [0.2, -0.5, 0.0, -3.14, -3.14, -3.14]
ACTION_MAX = [0.8,  0.5, 0.8,  3.14,  3.14,  3.14]
```

---

### 3.4 QNetwork (Twin Q-Critic) — подробно

**Файл**: `train_odin_sac_rl.py`, класс `QNetwork`

```python
QNetwork(obs_dim=274, action_dim=6, hidden_dim=256)
```

**Режим с scene embedding** (`obs_dim >= 274`):

```
obs (274,) разбивается на:
  curr_pos     = obs[:3]         # позиция камеры
  curr_orn     = obs[3:7]        # ориентация
  delta_p      = obs[7:8]        # изменение покрытия
  joints       = obs[8:15]       # суставы
  nbv_hint     = obs[15:18]      # подсказка ODIN
  scene_emb    = obs[18:274]     # semantic embedding (256,)

action (6,) разбивается на:
  target_pos   = action[:3]
  target_euler = action[3:]

Вычисляются геометрические фичи:
  rel_pos    = target_pos - curr_pos         # (3,)
  rel_euler  = target_euler - curr_euler     # (3,)
  dist       = ||rel_pos||                   # (1,)
  direction  = rel_pos / (dist + ε)         # (3,)
  hint_rel   = target_pos - nbv_hint        # (3,) отклонение от NBV-совета ODIN
  hint_dist  = ||hint_rel||                 # (1,)
  geom_feats = cat([rel_pos, rel_euler, dist, direction, hint_rel, hint_dist])  # (14,)

OOB фичи (насколько action выходит за bounds):
  oob_low  = ReLU(ACTION_MIN - target_pos)  # (3,)
  oob_high = ReLU(target_pos - ACTION_MAX)  # (3,)
  oob_feats = cat([oob_low, oob_high])       # (6,)

scene_feats = scene_proj(scene_emb)          # 256 → 128 → 64 (ReLU)

x = cat([scene_feats(64), geom_feats(14), oob_feats(6), delta_p(1), joints(7)])  # (92,)
Q = MLP(x)  # 92 → 256 → 256 → 1
```

**Два независимых пути** (twin):
- `q1_mlp` + `scene_proj1` → Q1
- `q2_mlp` + `scene_proj2` → Q2

**Режим без scene embedding** (`obs_dim < 274`):  
Простой `cat(obs, action) → MLP`

---

### 3.5 Actor (политика)

**НЕ отдельная сеть!** Актор — это NBV Head самой модели ODIN.

```python
# Online inference (реальное действие):
result = adapter.infer_for_rl(current_pos, current_quat)
nbv_pos = result["nbv_pos_t"]    # (3,) tensor С градиентами
nbv_quat = result["nbv_quat_t"]  # (4,) tensor С градиентами
euler = quat_to_euler_torch(nbv_quat)  # (3,)
mean_action = cat([nbv_pos, euler])    # (6,)

std = exp(log_std).clamp(1e-4, 2.0)
action = Normal(mean_action, std).rsample()  # reparameterization

# Surrogate actor для обновлений из replay buffer:
nbv_hint = obs_vec[:, 15:18]   # (B, 3) — уже записан в буфер
euler_hint = obs_vec[:, 3:6] * 0.1
mean = cat([nbv_hint, euler_hint])  # (B, 6)
action = Normal(mean, std).rsample()
```

> ⚠️ **КРИТИЧЕСКИЙ АРХИТЕКТУРНЫЙ ИЗЪЯН**: surrogate actor ≠ реальный actor.  
> При обновлении из replay buffer используется `nbv_hint` (прошлая подсказка ODIN),  
> а не реальный NBV Head с backbone. Это off-policy mismatch.

---

### 3.6 Reward Function

Вычисляется в двух местах:

**В `environment_odin.py`** (на каждом шаге `step()`):
```
if collision:
    reward = PENALTY_COLLISION = -15.0
elif out_of_bounds:
    reward = PENALTY_OOB = -10.0
else:
    rew_classifier   = Δmean_classifier_conf * REWARD_CLASSIFIER_SCALE (30.0)
    rew_all_found    = REWARD_ALL_FOUND_BONUS (50.0) если нашли ВСЕ объекты
    rew_success      = REWARD_SUCCESS_CLASSIFIED_BONUS (100.0) если всё верно классифицировано
    rew_survival     = REWARD_SURVIVAL (1.0) за выживание
    reward = rew_classifier + rew_all_found + rew_success + rew_survival
```

**В `train_odin_sac_rl.py`** (дополнительно к env reward):
```
delta_p_hidden = p_hidden_before - p_hidden_after

if безопасный шаг (reward > PENALTY_OOB/2):
    if p_hidden < REWARD_EXPLORATION_THRESHOLD (0.05):
        reward += REWARD_EXPLORATION_COMPLETE_BONUS (20.0)
    else:
        reward += (1.0 - p_hidden) * REWARD_EXPLORATION_PARTIAL_FACTOR (2.0)
```

**Coverage Head** отдельно обучается BCE-лоссом vs GT:
```
loss_cov = BCE(p_hidden_logit, GT_p_hidden)
GT = 1.0 если есть скрытые объекты, иначе 0.0
```

---

### 3.7 ODINAdapter

**Файл**: `src/vision/odin_adapter.py`

Два режима инференса:

1. **`infer(pos, quat)`** — для eval/среды, модель в `.eval()`, без градиентов
2. **`infer_for_rl(pos, quat)`** — для обучения, модель в `.train()`, с градиентами для NBV Head

Буфер кадров:
- `num_frames=5` кадров (скользящее окно)
- Каждый кадр: RGB (3, H, W) тензор + depth + pose 4x4 + intrinsics
- Backprojection: `odin.modeling.backproject.backproject_dataloader`

Извлечение scene_embedding:
```python
# Из query_features последнего forward-прохода
q_feats = base_model._last_outputs["query_features"]  # (B, Q, 256)
scene_emb = q_feats.mean(dim=1).squeeze(0)            # (256,)
```

---

### 3.8 Заморозка backbone (флаги)

| Флаг | Обучается |
|------|-----------|
| (нет флагов) | Все параметры |
| `--freeze_backbone` | NBV Head + Coverage Head |
| `--freeze_backbone --train_last_transformer_block` | NBV Head + Coverage Head + последний блок трансформерного декодера |

Последний блок трансформера определяется автоматически по максимальному индексу в `transformer_self_attention_layers`, `transformer_cross_attention_layers`, `transformer_ffn_layers`.

---

### 3.9 Сохранение чекпоинтов

```python
state = {
    "nbv_head": {k:v for k,v in model.state_dict().items() if "nbv_head" in k or "coverage_head" in k},
    "critic": agent.critic.state_dict(),
    "log_std": agent.log_std.data,
    "log_alpha": agent.log_alpha.data,
}
```

Backbone НЕ сохраняется (предполагается повторная загрузка).

---

## 4. Конфигурация (`config.py`) — все ключевые параметры

```python
# Сцена
SCENE_STAGE = 3          # 1=simple, 2=multi-obj, 3=multi-obj+obstacles
MIN_OBSTACLES = 3
MAX_OBSTACLES = 7
MIN_OBJECTS = 2
MAX_OBJECTS = 10
MAX_STEPS_PER_EPISODE = 5   # ← КРИТИЧЕСКИ МАЛО

# Обучение
LEARNING_RATE = 1e-2        # ← КРИТИЧЕСКИ МНОГО для SAC
BUFFER_SIZE = 10000
BATCH_SIZE = 128
LEARNING_STARTS = 1000
IMAGE_SIZE = 256

# Награды
REWARD_SCALE = 20.0
PENALTY_OOB = -10.0
PENALTY_COLLISION = -15.0
REWARD_CLASSIFIER_SCALE = 30.0
REWARD_ALL_FOUND_BONUS = 50.0
REWARD_SUCCESS_CLASSIFIED_BONUS = 100.0
REWARD_SURVIVAL = 1.0
REWARD_EXPLORATION_THRESHOLD = 0.05
REWARD_EXPLORATION_COMPLETE_BONUS = 20.0
REWARD_EXPLORATION_PARTIAL_FACTOR = 2.0
```

---

## 5. Структура файлов

```
NBV_with_obstacles_and_robot/
├── config.py                         # Все гиперпараметры
├── train_odin_sac_rl.py              # SAC обучение (главный скрипт)
├── train_odin_nbv_rl.py              # REINFORCE вариант (старый)
├── src/
│   ├── simulation/
│   │   └── environment_odin.py       # Gymnasium среда NBVODINEnv
│   ├── vision/
│   │   └── odin_adapter.py           # ODINAdapter + load_nbv_active_odin()
│   ├── data/
│   │   └── objects/                  # Объекты сцены
│   └── utils/
│       └── math_utils.py
├── tests/
│   ├── test_odin_adapter.py          # Тесты среды и адаптера (44 passed)
│   ├── test_odin_sac_agent.py        # Тесты SAC агента
│   └── test_odin_nbv_agent.py        # Тесты NBV head
├── notebooks/
│   └── kaggle_odin_rl_train.ipynb    # Ноутбук для Kaggle
└── kaggle_output/                    # Результаты обучения
    └── output_odin_sac/
        └── sac_metrics.csv
```

---

## 6. Результаты наблюдаемого обучения

> Данные: ~900 эпизодов из `kaggle_output/output_odin_sac/sac_metrics.csv`

| Метрика | Значение |
|---------|----------|
| Collision rate | **~79%** |
| OOB rate | **~7.8%** |
| Success rate | **0%** |
| Avg episode reward | отрицательный |
| p_hidden trend | не снижается |

---

## 7. Анализ причин плохого обучения

### 🔴 Причина 1: MAX_STEPS_PER_EPISODE = 5 — слишком мало

За 5 шагов агент не успевает получить положительный опыт. Почти каждый эпизод заканчивается только штрафами. В replay buffer накапливается преимущественно негативный сигнал → Q-функция учится только предсказывать плохое, но не хорошее.

**Рекомендация**: `MAX_STEPS_PER_EPISODE = 20-50`

---

### 🔴 Причина 2: LEARNING_RATE = 1e-2 — в 30 раз слишком большой для SAC

SAC стандартно обучается с `lr ≈ 3e-4`. При `1e-2`:
- Q-функция расходится (loss взрывается или уходит в noisy plateau)
- Политика не может стабилизироваться
- alpha (entropy коэффициент) может уйти в крайние значения

Значение `LEARNING_RATE = 1e-2` в `config.py` — это **legacy** от других экспериментов. В `train_odin_sac_rl.py` используются аргументы `--lr_actor=1e-4`, `--lr_critic=3e-4` — они правильные. Но `config.py` содержит старое значение, которое может переопределяться где-то ещё.

**Рекомендация**: `config.LEARNING_RATE = 3e-4` и проверить, где это значение читается.

---

### 🔴 Причина 3: SCENE_STAGE = 3 слишком сложно для старта

3-7 препятствий + 2-10 объектов в маленьком боксе:
```
X: [0.2, 0.8]  →  0.6м
Y: [-0.3, 0.3] →  0.6м  
Z: [0.15, 0.4] →  0.25м
```

Геометрически: 7 препятствий (scale 1.5) занимают значительную часть ~0.09 м³ объёма. При случайном действии вероятность попасть в препятствие → высокая. **Это объясняет 79% коллизий** — это почти чисто геометрическая вероятность, а не поведение агента.

**Рекомендация**:
1. Начать с `SCENE_STAGE = 1` (1 объект, нет препятствий)
2. Curriculum: перейти на Stage 2 после N эпизодов без коллизий
3. Перейти на Stage 3 только при стабильном обучении

---

### 🔴 Причина 4: Surrogate actor — структурная проблема off-policy mismatch

В `get_action_from_obs_vec()` (для обновлений critic/actor из replay buffer):
```python
# Суррогатный актор использует nbv_hint из старого obs
nbv_hint = obs_vec_batch[:, 15:18]   # прошлая подсказка ODIN
mean = cat([nbv_hint, euler_hint])
```

Реальный актор (online):
```python
# NBV Head ODIN + backbone inference
result = adapter.infer_for_rl(...)
mean_action = cat([nbv_pos, euler])
```

Это **разные функции**, но SAC предполагает что они одинаковы. Политический градиент считается неправильно → обучение нестабильно.

**Рекомендация**: Переключиться на **PPO** (on-policy), где surrogate actor = реальный actor, или создать отдельный MLP-actor с теми же параметрами.

---

### 🔴 Причина 5: Coverage head сигнал vs reward — двойной счёт

`p_hidden` из Coverage Head используется **и как reward**, и Coverage Head **обновляется BCE-лоссом** на каждом шаге. Это значит:

1. Coverage Head быстро учится предсказывать GT (есть скрытые/нет)
2. Когда Coverage Head точный — p_hidden ≈ 0 почти всегда (агент ничего не видит → GT=1 → p_hidden→1)
3. delta_p_hidden ≈ 0 почти всегда → reward_coverage ≈ 0

**Coverage reward не работает** при правильно обученном Coverage Head в режиме Stage 3 с мало шагов.

---

### 🟡 Причина 6: obs_vec (18) vs replay buffer (274) несоответствие

В replay buffer хранится `obs_vec_full (274)`. Surrogate actor читает:
```python
nbv_hint = obs_vec_batch[:, 15:18]  # ok, это правильные индексы для 18-dim
```
Но `obs_vec_batch` имеет размерность 274 (18 + 256). Индексы `[15:18]` совпадают → **работает правильно**. Это не баг, но архитектурно неочевидно.

---

## 8. Критические рекомендации для следующего агента

### Что изменить немедленно (обязательно):

```python
# config.py
MAX_STEPS_PER_EPISODE = 25     # было 5
SCENE_STAGE = 1                # начать с простого
MIN_OBJECTS = 1
MAX_OBJECTS = 3
```

```bash
# Запуск (аргументы уже правильные в train_odin_sac_rl.py):
python train_odin_sac_rl.py \
    --freeze_backbone \
    --scene_stage 1 \
    --max_steps 25 \
    --total_steps 200000 \
    --lr_actor 1e-4 \
    --lr_critic 3e-4 \
    --batch_size 64 \
    --learning_starts 500 \
    --buffer_size 50000
```

### Что требует архитектурного решения (важно):

1. **Surrogate actor**: Либо PPO, либо создать отдельный MLP actor с 18/274 → 6 и использовать его везде
2. **Coverage reward**: Отвязать Coverage Head от RL reward или не обучать Coverage Head во время RL
3. **Curriculum**: Добавить автоматический переход между Stage 1 → 2 → 3 по метрикам

### Что работает правильно (не трогать):

- QNetwork с геометрическими фичами — архитектурно умный, правильный
- `--train_last_transformer_block` флаг — реализован корректно
- ODINAdapter.infer() / infer_for_rl() — разделение eval/train режимов правильное
- Reward структура в `config.py` — все параметры задокументированы, легко менять
- Тестовое покрытие — 44 тестов, все зелёные

---

## 9. Тестирование

```bash
python -m pytest tests/ -v
# 44 passed, 0 failed
```

Покрывает:
- `TestHelpers` — backprojection математика
- `TestODINAdapter` — буфер кадров, infer, пустые результаты
- `TestNBVODINEnvNoAdapter` — среда без ODIN
- `TestNBVODINEnvWithAdapter` — среда с мок-адаптером, collision/OOB/reward
- `TestReplayBuffer` — push/sample/capacity
- `TestQNetwork` — forward shapes, gradient flow
- `TestSACTrainingLoop` — critic update, actor update, backbone freeze

---

## 10. Мой итоговый вердикт

> **Архитектура концептуально верная, но не дотюнена до рабочего состояния.**

Идея использовать ODIN NBV Head как RL-политику — умная. scene_embedding из query_features трансформера — богатое представление сцены. QNetwork с геометрическими фичами — на уровне современных work.

Но **три простые ошибки** делают обучение нерабочим:
1. `MAX_STEPS_PER_EPISODE = 5` — слишком короткие эпизоды
2. `SCENE_STAGE = 3` — слишком сложная среда для старта
3. Surrogate actor ≠ реальный actor → gradient mismatch

Ни одна из этих проблем не требует переписывания архитектуры.  
Всё решается правкой конфига и небольшой переработкой `get_action_from_obs_vec()`.

**Ожидаемый результат после исправлений**: collision rate < 30%, первые успехи на Stage 1 через ~50к шагов.
