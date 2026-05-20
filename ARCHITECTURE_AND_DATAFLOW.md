# Полная архитектура и Data Flow: NBV RL с ODIN

> Документ описывает всё что происходит при обучении (SAC и PPO) и инференсе.  
> Читай сверху вниз: сначала архитектура всех компонентов, потом потоки данных, потом формулы.

---

## Оглавление

1. [Общая идея](#1-общая-идея)
2. [Пространство действий и наблюдений](#2-пространство-действий-и-наблюдений)
3. [Архитектуры нейросетей](#3-архитектуры-нейросетей)
   - 3.1 [ODIN backbone (NBVActiveODIN)](#31-odin-backbone-nbvactiveodin)
   - 3.2 [Coverage Head](#32-coverage-head)
   - 3.3 [NBV Head](#33-nbv-head)
   - 3.4 [ActorMLP (SAC)](#34-actormlp-sac)
   - 3.5 [QNetwork — Twin Q-critics (SAC)](#35-qnetwork--twin-q-critics-sac)
   - 3.6 [PPOActorCritic (PPO)](#36-ppoactorcritic-ppo)
4. [Функция награды](#4-функция-награды)
5. [Функции потерь](#5-функции-потерь)
6. [Data Flow: SAC обучение](#6-data-flow-sac-обучение)
7. [Data Flow: PPO обучение](#7-data-flow-ppo-обучение)
8. [Data Flow: Инференс](#8-data-flow-инференс)
9. [Что обучается, что заморожено и почему](#9-что-обучается-что-заморожено-и-почему)
10. [Конфигурация гиперпараметров](#10-конфигурация-гиперпараметров)

---

## 1. Общая идея

Задача — обучить агента управлять камерой (закреплённой на роботе-манипуляторе Kuka iiwa) так, чтобы за ограниченное число шагов:
1. Найти все целевые объекты в сцене.
2. Правильно классифицировать их (определить класс).
3. Минимизировать долю «скрытых» (не обнаруженных) объектов — `p_hidden`.

Агент управляет **абсолютными координатами** следующей позиции камеры `[x, y, z, roll, pitch, yaw]`.  
Визуальным «мозгом» служит модель **ODIN** — 3D-паноптический сегментатор, обученный распознавать объекты из последовательности RGB-D кадров.

```
PyBullet сцена
      │  RGB-D кадр (256×256×4)
      ▼
  ODINAdapter
      │  scene_emb (256,)  ←── query features ODIN
      │  p_hidden  ←── Coverage Head
      │  instances_3d  ←── сегментация 3D
      ▼
  RL агент (SAC или PPO)
      │  action: [x, y, z, roll, pitch, yaw]
      ▼
  Robot.apply_action(pos, orn)
      │  IK → целевые углы суставов
      ▼
  Следующий кадр
```

---

## 2. Пространство действий и наблюдений

### Action Space

Вектор 6D — абсолютные координаты целевой позиции камеры в пространстве:

| Компонент | Мин | Макс | Смысл |
|---|---|---|---|
| x | 0.2 | 0.8 | Глубина (вперёд) |
| y | −0.5 | 0.5 | Боковое смещение |
| z | 0.0 | 0.8 | Высота |
| roll | −π | π | Вращение вокруг X |
| pitch | −π | π | Вращение вокруг Y |
| yaw | −π | π | Вращение вокруг Z |

Почему абсолютные, а не дельты? Потому что ODIN NBV Head предсказывает «лучшую следующую точку» в мировых координатах, и reward напрямую зависит от достигнутой позиции.

### Observation Space

Среда возвращает словарь:

```python
obs = {
    "image": np.float32,  # shape (4, 256, 256) — нормализованный RGBD
    "vector": np.float32, # shape (18,) — структурированный вектор
}
```

**Структура `obs["vector"]` (18 компонентов):**

```
[0:3]   — pos (3)       — текущая позиция EE/камеры [x, y, z]
[3:7]   — orn (4)       — ориентация в виде кватерниона [x, y, z, w]
[7]     — delta_p (1)   — delta_p_hidden за последний шаг (p_prev - p_curr)
[8:15]  — joints (7)    — углы 7 суставов робота
[15:18] — nbv_hint (3)  — рекомендуемая позиция от ODIN NBV Head (x, y, z)
```

**Расширенный вектор для RL агентов:**

Перед подачей в ActorMLP / PPOActorCritic к `obs["vector"]` конкатенируется `scene_emb`:

$$\mathbf{obs\_vec\_full}_{(274)} = \mathbf{obs["vector"]}_{(18)} \oplus \mathbf{scene\_emb}_{(256)}$$

`scene_emb` — это среднее по запросам `query_features` ODIN трансформера, сжатое пространственное описание сцены.

---

## 3. Архитектуры нейросетей

### 3.1 ODIN backbone (NBVActiveODIN)

ODIN — это 3D-адаптация архитектуры **Mask2Former** для паноптической сегментации по последовательности RGB-D кадров.

#### Входные данные

```
images:     (N, 3, H, W)     — N≤5 RGB кадров, float32
depths:     (N, H, W)        — глубина в метрах
poses:      (N, 4, 4)        — матрицы camera-to-world
intrinsics: (N, 3, 3)        — матрицы интринзики K
```

где N = количество накопленных кадров (скользящее окно глубиной 5).

#### Структура

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │                      ODIN Base Model                                 │
 │                                                                      │
 │  images[n]  ──→  2D ResNet Backbone  ──→  Feature Pyramid  F        │
 │                                                   │                  │
 │  depth[n] + pose[n]  ──→  Backprojector  ──→  3D XYZ (multi-scale) │
 │    (depth→XYZ в camera coords → world coords)     │                  │
 │                                    ┌──────────────┘                  │
 │                                    │  F + XYZ                        │
 │                                    ▼                                 │
 │                   ┌─────────────────────────────┐                   │
 │                   │  Pixel Decoder (FPN-style)  │                   │
 │                   │  Multi-scale features fused │                   │
 │                   └──────────────┬──────────────┘                   │
 │                                  │                                   │
 │  Object queries Q ∈ R^(num_queries × 256)  ←  learnable             │
 │                                  │                                   │
 │                   ┌──────────────▼──────────────────────┐           │
 │                   │    Transformer Decoder (L блоков)    │           │
 │                   │                                      │           │
 │                   │  Блок k:                             │           │
 │                   │    Self-Attention(Q, Q)              │           │
 │                   │    Cross-Attention(Q, F)  ← image   │           │
 │                   │    Cross-Attention-3D(Q, XYZ) ← 3D  │           │
 │                   │    FFN(Q)                            │           │
 │                   │                                      │           │
 │                   └──────────────┬──────────────────────┘           │
 │                                  │                                   │
 │             query_features  [B, Q, 256]                              │
 │             сохраняется в _last_outputs["query_features"]            │
 │                                  │                                   │
 │              ┌───────────────────┤                                   │
 │              ▼                   ▼                                   │
 │          Mask Head           Class Head                              │
 │          pred_masks          pred_classes                            │
 │          [B, Q, H, W]        [B, Q, num_classes]                    │
 └──────────────────────────────────────────────────────────────────────┘
                                    │
              query_features [B, Q, 256]  ──→  передаётся наружу
                      │                         в NBVActiveODIN
              ┌───────┴───────────────────────────┐
              ▼                                   ▼
 ┌────────────────────────┐          ┌─────────────────────────────┐
 │     Coverage Head      │          │         NBV Head            │
 │  (скрытые объекты?)    │          │  (куда двигать камеру?)     │
 │                        │          │                             │
 │  flatten(Q,D)          │          │  flatten(Q,D)               │
 │  [B, 5120]             │          │  [B, 5120] = flat_feats     │
 │  Linear(5120→128)+ReLU │          │  concat(flat_feats,         │
 │  Linear(128→1)         │          │         curr_pos[3],        │
 │       │                │          │         curr_quat[4])       │
 │  logit [B, 1]          │          │  → [B, 5127]                │
 │  σ(logit) = p_hidden   │          │  Linear(5127→128)+LN+ReLU   │
 │  ∈ [0, 1]              │          │  Linear(128→128)+LN+ReLU    │
 │                        │          │  Linear(128→64)+ReLU        │
 │  p_hidden → 1 :        │          │         │                   │
 │    много скрытых        │          │    ┌────┴────┐             │
 │  p_hidden → 0 :        │          │    ▼         ▼             │
 │    сцена исследована    │          │  pos_head  quat_head        │
 │                        │          │  Lin(64,3) Lin(64,4)        │
 └────────────────────────┘          │  nbv_pos   normalize()      │
          │                          │  [B, 3]    nbv_quat [B, 4]  │
          │                          └─────────────────────────────┘
          │                                       │
          ▼                                       ▼
   p_hidden (float)                   рекомендованная поза камеры
   используется для:                  используется как:
   • rew_coverage (reward)            • behavioral policy в SAC
   • exploration_bonus                • nbv_hint[3] в obs_vec[15:18]
   • обучается BCE loss               • обучается через actor_optimizer
```

#### Выходные данные base model

- `instances_3d`: словарь с `pred_masks`, `pred_classes`, `pred_scores`
- `query_features`: `[B, Q, 256]` — главный интерфейс для Coverage Head и NBV Head

**Backprojection** (геометрическая операция, без градиентов):

$$X_{cam} = \frac{u - c_x}{f_x} \cdot d, \quad Y_{cam} = -\frac{v - c_y}{f_y} \cdot d, \quad Z_{cam} = -d$$

$$\mathbf{X}_{world} = R \cdot \mathbf{X}_{cam} + \mathbf{t}$$

Конвенция: OpenGL (+Y вверх, −Z вперёд) → ODIN (+Z вперёд при backprojection flip).

---

### 3.2 Coverage Head

**Задача:** Предсказать, есть ли ещё скрытые объекты в сцене ($p_{hidden} \in [0, 1]$).

**Входные данные:** `query_features [B, Q, 256]`

**Архитектура:**

```
query_features [B, Q=20, 256]
       │
  flatten(Q, D)            # Q=20, D=256 → сохраняем всю per-query информацию
       │
  [B, 5120]
       │
  Linear(5120, 128) + ReLU
       │
  [B, 128]
       │
  Linear(128, 1)
       │
  pred_logits [B, 1]       # сырой логит (без сигмоиды, для AMP-стабильности)
       │
  sigmoid(logits)          # только при инференсе → p_hidden ∈ [0, 1]
```

**Инициализация:** Xavier uniform.

**Выход:** $p_{hidden} = \sigma(\ell)$ — вероятность того, что в сцене есть скрытые объекты.  
$p_{hidden} \to 1$ = не видели ещё ничего.  
$p_{hidden} \to 0$ = сцена полностью исследована.

---

### 3.3 NBV Head

**Задача:** Предсказать, куда переместить камеру для максимального раскрытия сцены.

**Входные данные:**
- `query_features [B, Q, 256]`
- `current_position [B, 3]`
- `current_quaternion [B, 4]`

**Архитектура:**

```
query_features [B, Q=20, 256]
       │
  flatten(Q, D)  →  flat_feats [B, 5120]    # сохраняем всю per-query информацию
       │
  concat([flat_feats, current_pos, current_quat])
       │
  [B, 5127]  (5120 + 3 + 4)
       │
  Linear(5127, 128) + LayerNorm + ReLU
       │
  Linear(128, 128) + LayerNorm + ReLU
       │
  Linear(128, 64) + ReLU
       │
  [B, 64]
       ├─────────────────────┐
       ▼                     ▼
  Linear(64, 3)         Linear(64, 4)
  pos_head              quat_head
       │                     │
  nbv_pos [B, 3]        normalize(·) → nbv_quat [B, 4]
  (мировые coord.)      (единичный кватернион)
```

**Выход:** `(nbv_pos, nbv_quat)` — рекомендованная следующая позиция и ориентация камеры.

В SAC NBV Head используется как **поведенческая политика** при сборе данных (exploration).  
В PPO NBV Head **не используется как политика** — только как источник `nbv_hint` в obs_vec.

---

### 3.4 ActorMLP (SAC)

**Задача:** Единая политика SAC — и behavioral (сбор данных), и learned (апдейты).  
Это стандартный SAC без mismatch: Q обучается на ActorMLP-действиях, актор максимизирует Q в той же области.  
ODIN NBV Head при этом — источник `nbv_hint` в $v_t[15{:}18]$, доступного ActorMLP как входной признак.

**Входные данные:** `obs_vec_full [B, 274]`

**Архитектура:**

```
obs_vec_full [B, 274]
       │
  Linear(274, 256) + LayerNorm + ReLU
       │
  Linear(256, 256) + LayerNorm + ReLU
       │
  Linear(256, 6)
       │
  mean_action [B, 6]
```

Стохастическое действие через reparameterization trick:

$$a = \mu_\theta(s) + \varepsilon \cdot \sigma_\theta(s), \quad \varepsilon \sim \mathcal{N}(0, I)$$

$$\log \pi(a|s) = \sum_i \log \mathcal{N}(a_i \mid \mu_i, \sigma_i)$$

$\log\sigma$ — обучаемый параметр $[6]$, shared для всего батча.  
$\sigma = \text{clamp}(\exp(\log\sigma),\; 1{{\times}}10^{-4},\; 2.0)$.

**Выход:** `(action [B, 6], log_prob [B, 1])`

---

### 3.5 QNetwork — Twin Q-critics (SAC)

**Задача:** Оценить $Q(s, a)$ = ожидаемый кумулятивный дисконтированный reward.

**Входные данные:** `obs [B, 274]`, `action [B, 6]`

**Архитектура (для каждого из двух Q-networks):**

```
obs [B, 274]:
  scene_emb = obs[:, 18:274]    → Linear(256,128)+ReLU → Linear(128,64)+ReLU → [B, 64]
  curr_pos  = obs[:, 0:3]
  delta_p   = obs[:, 7:8]
  joints    = obs[:, 8:15]
  nbv_hint  = obs[:, 15:18]

action [B, 6]:
  target_pos  = action[:, 0:3]
  target_euler = action[:, 3:6]

Геометрические признаки (14):
  rel_pos    = target_pos - curr_pos          # [3]
  curr_euler = obs[:, 3:6] * 0.1             # грубая оценка из orn
  rel_euler  = target_euler - curr_euler      # [3]
  dist       = ||rel_pos||                    # [1]
  direction  = rel_pos / (dist + ε)           # [3]
  hint_rel   = target_pos - nbv_hint          # [3]
  hint_dist  = ||hint_rel||                   # [1]
  → concat → [14]

OOB признаки (6):
  oob_low  = ReLU([0.2,-0.5,0.05] - target_pos)   # [3]
  oob_high = ReLU(target_pos - [0.8,0.5,0.8])      # [3]
  → concat → [6]

x = concat(scene_feats[64], geom[14], oob[6], delta_p[1], joints[7])
  = [B, 92]
       │
  Linear(92, 256) + ReLU
  Linear(256, 256) + ReLU
  Linear(256, 1)
       │
  Q_value [B, 1]
```

Twin Q: два независимых MLP с отдельными проекциями scene_emb (`scene_proj1`, `scene_proj2`).  
Target network: soft-copy с $\tau = 0.005$.

---

### 3.6 PPOActorCritic (PPO)

**Задача:** Единая архитектура актора и критика для on-policy PPO.

**Входные данные:** `obs_vec_full [B, 274]`

**Архитектура:**

```
obs_vec_full [B, 274]
       │
  ┌────┤ Shared Trunk ├────┐
  │                        │
  Linear(274, 256) + LayerNorm + Tanh
  Linear(256, 256) + LayerNorm + Tanh
       │
  features [B, 256]
       ├─────────────────────┐
       ▼                     ▼
  actor_head             critic_head
  Linear(256, 6)         Linear(256, 1)
       │                     │
  mean_action [B, 6]     value [B, 1]
```

Стохастическая политика и энтропия:

$$\log \pi(a|s) = \sum_i \log \mathcal{N}(a_i \mid \mu_i, \sigma_i)$$

$$\mathcal{H}[\pi(\cdot|s)] = \sum_i \left(\frac{1}{2} + \log\!\left(\sigma_i \sqrt{2\pi e}\right)\right)$$

`actor_log_std` — обучаемый параметр $[6]$.

**Инициализация:** orthogonal init.
- Trunk layers: $\text{gain} = \sqrt{2}$
- actor_head: $\text{gain} = 0.01$ (малые веса → начальная политика близка к равномерной)
- critic_head: $\text{gain} = 1.0$

**Выход:** `(action, log_prob, entropy, value)`

---

## 4. Функция награды

Награда вычисляется в `train_odin_sac_rl.py` / `train_odin_ppo_rl.py` после каждого шага.

### 4.1 Структура

$$r_t = r_{env} + r_{coverage} \cdot \mathbf{1}_{safe} + r_{exploration} \cdot \mathbf{1}_{safe}$$

**Признак безопасного шага:**  
`is_safe = (env_reward > max(PENALTY_OOB, PENALTY_COLLISION) / 2)`

---

### 4.2 Env Reward (компонент из среды)

Среда `NBVODINEnv.step()` возвращает $r_{env}$:

$$r_{env} = \begin{cases} -15 & \text{коллизия} \\ -10 & \text{OOB} \\ \Delta_{conf} \cdot 30 + r_{all\_found} + r_{classified} + 1.0 & \text{иначе} \end{cases}$$

где:
- $\Delta_{conf} = \bar{c}_{t} - \bar{c}_{t-1}$ — изменение средней уверенности классификатора ODIN
- $r_{all\_found} = 50.0$ — однократный бонус при первом обнаружении всех объектов
- $r_{classified} = 100.0$ — однократный бонус за правильную классификацию всех объектов (завершает эпизод)

---

### 4.3 Coverage Reward

$$\Delta p_{hidden} = p_{hidden}^{(t)} - p_{hidden}^{(t+1)}$$

$$r_{coverage} = \Delta p_{hidden} \times 20.0$$

$\Delta p_{hidden} > 0$ означает, что скрытость сцены снизилась → положительный reward.  
$p_{hidden}^{(t+1)}$ получается отдельным вызовом `adapter.infer_for_rl(next_obs)`.

**Важно:** это был баг в исходном коде — `rew_coverage` не попадал в replay buffer. Исправлено.

---

### 4.4 Exploration Bonus

$$r_{exploration} = \begin{cases} 20.0 & \text{если } p_{hidden} < 0.05 \\ (1 - p_{hidden}) \times 2.0 & \text{иначе} \end{cases}$$

Обеспечивает непрерывный градиент: чем ниже $p_{hidden}$, тем выше бонус.

---

### 4.5 Итоговая формула

$$r_t = \begin{cases} -15 & \text{коллизия} \\ -10 & \text{OOB} \\ r_{env} + (p_{hidden}^{(t)} - p_{hidden}^{(t+1)}) \cdot 20 + r_{exploration} & \text{иначе} \end{cases}$$

---

## 5. Функции потерь

### 5.1 Coverage Head — BCE Loss

Обучается независимо от RL на каждом шаге (supervised, self-labeled из среды).

$$y = \begin{cases} 1.0 & \text{если } n_{hidden} > 0 \\ 0.0 & \text{иначе} \end{cases}$$

$$\mathcal{L}_{cov} = -\Bigl[y \cdot \log\sigma(\ell) + (1-y)\cdot\log(1-\sigma(\ell))\Bigr]$$

Градиент течёт **только** через Coverage Head.  
$p_{hidden}$, используемый в reward, берётся как `float` (`.detach()`) → нет утечки градиента в reward.

---

### 5.2 SAC — Critic Loss

Twin Q-networks обучаются минимизировать Bellman error:

$$y_t = r_t + \gamma(1 - d_t)\cdot\Bigl[\min\!\bigl(Q_1^{tgt}, Q_2^{tgt}\bigr)(s_{t+1}, \tilde{a}_{t+1}) - \alpha\cdot\log\pi(\tilde{a}_{t+1}|s_{t+1})\Bigr]$$

$$\tilde{a}_{t+1} \sim \text{ActorMLP}(s_{t+1}) \quad \text{(reparameterization)}$$

$$\mathcal{L}_{critic} = \text{MSE}\!\bigl(Q_1(s_t, a_t),\; y_t\bigr) + \text{MSE}\!\bigl(Q_2(s_t, a_t),\; y_t\bigr)$$

Параметры: $\gamma = 0.99$, $\tau = 0.005$ (soft target update).

---

### 5.3 SAC — Actor Loss

Максимизирует ожидаемый Q с entropy regularization:

$$\mathcal{L}_{actor} = \mathbb{E}_{s \sim \mathcal{D},\;\tilde{a} \sim \pi}\Bigl[\alpha\cdot\log\pi(\tilde{a}|s) - \min\!\bigl(Q_1(s,\tilde{a}),\;Q_2(s,\tilde{a})\bigr)\Bigr]$$

$\tilde{a}$ семплируется через reparameterization trick: $\tilde{a} = \mu_\theta(s) + \varepsilon\cdot\sigma_\theta(s),\;\varepsilon \sim \mathcal{N}(0,I)$

---

### 5.4 SAC — Entropy Coefficient Loss

Автоматическое управление температурой (auto entropy tuning):

$$H_{target} = -|\mathcal{A}| = -6$$

$$\mathcal{L}_\alpha = -\log\alpha \cdot \underbrace{\bigl(\log\pi(\tilde{a}|s) + H_{target}\bigr)}_{\text{stop\_grad}}$$

$\log\alpha$ — обучаемый скаляр. Оптимизируется Adam с тем же `lr_actor`.

---

### 5.5 PPO — Policy Loss (Clipped Surrogate)

$$\rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} = \exp\!\Bigl(\log\pi_\theta(a_t|s_t) - \log\pi_{\theta_{old}}(a_t|s_t)\Bigr)$$

$$\mathcal{L}_{PG} = -\mathbb{E}_t\Bigl[\min\!\Bigl(\rho_t\,\hat{A}_t,\;\;\text{clip}(\rho_t,\,1{-}\varepsilon,\,1{+}\varepsilon)\cdot\hat{A}_t\Bigr)\Bigr], \quad \varepsilon = 0.2$$

---

### 5.6 PPO — Value Loss

$$\mathcal{L}_V = \text{MSE}\!\bigl(V_\theta(s_t),\;R_t\bigr), \quad R_t = \hat{A}_t + V(s_t) \quad \text{(GAE returns)}$$

---

### 5.7 PPO — Entropy Bonus

$$\mathcal{L}_{ent} = -\mathcal{H}\bigl[\pi_\theta(\cdot|s_t)\bigr] = -\mathbb{E}_t\bigl[\text{entropy}(\text{dist})\bigr]$$

---

### 5.8 PPO — Total Loss

$$\mathcal{L}_{PPO} = \mathcal{L}_{PG} + 0.5\cdot\mathcal{L}_V - 0.01\cdot\mathcal{H}$$

Градиентный клип: $\|\nabla\|_2 \leq 0.5$.

---

### 5.9 GAE — Generalized Advantage Estimation (PPO)

$$\delta_t = r_t + \gamma\cdot V(s_{t+1})\cdot(1 - d_t) - V(s_t)$$

$$\hat{A}_t = \sum_{k=0}^{T-t}(\gamma\lambda)^k\,\delta_{t+k} = \delta_t + \gamma\lambda\cdot\hat{A}_{t+1} \quad \text{(рекуррентно, с конца)}$$

$$\gamma = 0.99, \quad \lambda = 0.95$$

---

## 6. Data Flow: SAC обучение

### 6.1 Схема одного шага

**1. ODIN inference — текущий obs**

$$z_t,\; \hat{p}_t,\; \ell_t \;\leftarrow\; \text{ODIN}(I_t,\, D_t,\, T_t)$$

$$v_t[15{:}18] \leftarrow \text{NBVHead\_pos}_t, \qquad s_t = [v_t \;\|\; z_t] \in \mathbb{R}^{274}$$

где $v_t \in \mathbb{R}^{18}$ — вектор среды; NBV Head позиция инжектируется в $v_t[15{:}18]$ как `nbv_hint` — входной признак для ActorMLP.

---

**2. Действие** (behavioral policy = learned policy = **ActorMLP**)

$$\mu_t = \text{ActorMLP}(s_t), \qquad \varepsilon \sim \mathcal{N}(0, I)$$

$$a_t = \text{clip}\!\left(\mu_t + \varepsilon \cdot \sigma_t,\; a_{\min},\; a_{\max}\right)$$

> Стандартный SAC: одна и та же сеть собирает данные и обновляется. Q обучается на ActorMLP-действиях — нет экстраполяции при actor update.

---

**3. Шаг среды**

$$(o_{t+1},\; r^{\text{env}}) \;\leftarrow\; \text{env.step}(a_t)$$

---

**4. ODIN inference — следующий obs**

$$z_{t+1},\; \hat{p}_{t+1} \;\leftarrow\; \text{ODIN}(o_{t+1}), \qquad s_{t+1} = [v_{t+1} \;\|\; z_{t+1}]$$

---

**5. Награда**

$$r_t = r^{\text{env}} + 20\cdot\underbrace{(\hat{p}_t - \hat{p}_{t+1})}_{\Delta p_{\text{hidden}}} \cdot \mathbf{1}[\text{safe}] + r^{\text{expl}}(\hat{p}_t) \cdot \mathbf{1}[\text{safe}]$$

---

**6. Coverage Head update** (supervised, изолировано от RL)

$$y_t = \mathbf{1}[\text{есть скрытые объекты}], \qquad \mathcal{L}_{\text{cov}} = \text{BCE}(\ell_t,\; y_t)$$

> $\ell_t$ — сырой логит из шага 1; при вычислении $r_t$ берётся $\hat{p}_t = \sigma(\ell_t)\texttt{.detach()}$ — градиент не течёт в RL.

---

**7. Replay buffer**

$$\mathcal{B} \;\leftarrow\; (s_t,\; a_t,\; r_t,\; s_{t+1},\; d_t)$$

---

**8. SAC update** (если $|\mathcal{B}| \geq N_{\text{start}}$, батч $(s, a, r, s', d) \sim \mathcal{B}$)

$$y^Q = r + \gamma(1-d)\Bigl[\min_i Q_i^{\text{tgt}}(s', \tilde{a}') - \alpha \log\pi(\tilde{a}'|s')\Bigr], \quad \tilde{a}' \sim \text{ActorMLP}(s')$$

$$\mathcal{L}_Q = \mathbb{E}\bigl[(Q_1(s,a) - y^Q)^2 + (Q_2(s,a) - y^Q)^2\bigr]$$

$$\mathcal{L}_\pi = \mathbb{E}\bigl[\alpha\log\pi(\tilde{a}|s) - \min_i Q_i(s, \tilde{a})\bigr]$$

$$\mathcal{L}_\alpha = \mathbb{E}\bigl[-\alpha\,(\log\pi(\tilde{a}|s) + \mathcal{H}_{\text{tgt}})\bigr], \qquad \theta^{\text{tgt}} \leftarrow \tau\,\theta + (1-\tau)\,\theta^{\text{tgt}}$$

### 6.2 Полный цикл SAC

```
while total_step < total_steps:
    obs = env.reset()
    adapter.reset()                # очистить frame buffer

    for step in range(MAX_STEPS=25):
        [шаг t — см. схему выше]

        if len(buffer) >= learning_starts(500):
            if total_step % update_freq(1) == 0:
                batch = buffer.sample(64)
                update_critic(batch)         → L_critic
                update_actor_and_alpha(batch) → L_actor, L_α
                soft_update_target()

        if done: break

    episode++
    log metrics to CSV
```

### 6.3 Update_critic детально

```python
obs, action, reward, next_obs, done = batch

# Target computation (no_grad)
next_action, next_log_prob = ActorMLP(next_obs) + log_std.sample()
q1_t, q2_t = Q_target(next_obs, next_action)
q_next = min(q1_t, q2_t) - α · next_log_prob
y = reward + γ · (1 - done) · q_next

# Q-networks
q1, q2 = Q(obs, action)          # stored actions из behavioral policy
L = MSE(q1, y) + MSE(q2, y)
critic_optimizer.step()
```

**Примечание:** `action` в batch и `next_action` в таргете — оба от **ActorMLP**.  
Стандартный SAC: Q обучается на актуальных действиях политики, actor update не экстраполирует Q за пределы обучающего распределения.

---

## 7. Data Flow: PPO обучение

### 7.1 Схема одного шага в rollout

**1. Coverage Head update** (до действия, на текущем $s_t$)

$$y_t = \mathbf{1}[\text{есть скрытые объекты}], \qquad \mathcal{L}_{\text{cov}} = \text{BCE}(\ell_t,\; y_t)$$

---

**2. Действие** (PPOActorCritic — та же сеть, что и при обновлении)

$$a_t,\; \log\pi_t,\; V_t \;\leftarrow\; \text{PPOActorCritic}(s_t)$$

$$a_t^{\text{clip}} = \text{clip}(a_t,\; a_{\min},\; a_{\max})$$

---

**3. Шаг среды**

$$(o_{t+1},\; r^{\text{env}}) \;\leftarrow\; \text{env.step}(a_t^{\text{clip}})$$

---

**4. ODIN inference — следующий obs** (без градиентов)

$$z_{t+1},\; \hat{p}_{t+1},\; \ell_{t+1} \;\leftarrow\; \text{ODIN}(o_{t+1})\big|_{\text{no\_grad}}$$

$$s_{t+1} = [v_{t+1} \;\|\; z_{t+1}]$$

---

**5. Награда**

$$r_t = r^{\text{env}} + 20\cdot(\hat{p}_t - \hat{p}_{t+1}) \cdot \mathbf{1}[\text{safe}] + r^{\text{expl}}(\hat{p}_t) \cdot \mathbf{1}[\text{safe}]$$

---

**6. Rollout buffer**

$$\mathcal{R} \;\leftarrow\; (s_t,\; a_t,\; r_t,\; V_t,\; \log\pi_t,\; d_t)$$

Состояние переходит: $s_t \leftarrow s_{t+1}$, $\ell_t \leftarrow \ell_{t+1}$.

### 7.2 Полный цикл PPO

```
obs = env.reset()
scene_emb, p_hidden, p_hidden_logit, instances_3d = ODIN(obs)
obs_vec_full = concat(obs["vector"], scene_emb)

while total_step < total_steps:
    rollout_buffer.clear()

    # ── Collect n_steps=512 шагов ──
    for _ in range(n_steps=512):
        [шаг t — см. схему выше]
        if done:
            [log episode metrics]
            obs = env.reset(); adapter.reset()
            obs_vec_full = ODIN(obs)...

    # ── PPO update ──
    last_value = PPOActorCritic.get_value(obs_vec_full)
    rollout_buffer.compute_gae(last_value, γ=0.99, λ=0.95)

    for epoch in range(n_epochs=10):
        for mini_batch in shuffle_and_split(buffer, batch_size=64):
            L = L_PG + 0.5·L_V + 0.01·L_ent
            optimizer.step()
            grad_clip(0.5)
```

### 7.3 Ключевое отличие PPO от SAC

| | SAC | PPO |
|---|---|---|
| Тип | Off-policy | On-policy |
| Буфер | Replay Buffer (случайные старые данные) | Rollout Buffer (только свежие данные) |
| Behavioral policy | ODIN NBV Head | PPOActorCritic |
| Learned policy | ActorMLP | PPOActorCritic (та же!) |
| Проблема mismatch | Была, исправлена ActorMLP | Нет по построению |
| Обновление | После каждого шага | После каждых 512 шагов |

---

## 8. Data Flow: Инференс

При инференсе запускается обученный агент без обновлений весов.

### 8.1 SAC Inference

```
obs = env.reset()
adapter.reset()

for step in range(MAX_STEPS):
    # 1. ODIN inference — scene_emb, p_hidden, nbv_hint
    adapter.add_frame(rgb, depth, pose, K)
    result = adapter.infer_for_rl(pos, quat)
    obs_vec[15:18] = result["nbv_pos_t"]          # inject nbv_hint
    obs_vec_full = concat(obs_vec, result["scene_emb"])

    # 2. ActorMLP deterministic action
    mean_action = ActorMLP(obs_vec_full)
    action = clip(mean_action, ACTION_MIN, ACTION_MAX)

    # 3. Шаг среды
    next_obs, reward, done, _ = env.step(action)

    if done: break
```

SAC инференс корректен: та же ActorMLP, что и при обучении, с `deterministic=True` (без шума).

### 8.2 PPO Inference

```
obs = env.reset()
adapter.reset()
scene_emb, p_hidden, _, _, _ = ODIN(obs)
obs_vec_full = concat(obs["vector"], scene_emb)

for step in range(MAX_STEPS):
    # 1. PPOActorCritic deterministic action
    h = trunk(obs_vec_full)
    mean_action = actor_head(h)
    action = clip(mean_action, ACTION_MIN, ACTION_MAX)

    # 2. Шаг среды
    next_obs, reward, done, _ = env.step(action)

    # 3. ODIN на следующий obs
    next_scene_emb, next_p_hidden, _, _, _ = ODIN(next_obs)
    obs_vec_full = concat(next_obs["vector"], next_scene_emb)

    if done: break
```

PPO-инференс корректен: та же PPOActorCritic, что и при обучении, просто без шума.

---

## 9. Что обучается, что заморожено и почему

### Режим `--freeze_backbone` (рекомендуется)

| Компонент | Статус | Почему |
|---|---|---|
| ODIN ResNet backbone | ❄️ Заморожен | Предобученный визуальный признаковый экстрактор. Обучение требует detectron2 и большого датасета. |
| ODIN Pixel Decoder | ❄️ Заморожен | То же |
| ODIN Transformer Decoder (все блоки) | ❄️ Заморожен | То же |
| Coverage Head | 🔥 Обучается (BCE) | Небольшая голова, self-labeled из среды |
| NBV Head (SAC) | ❄️ Заморожен | Только источник nbv_hint в obs_vec[15:18]; больше не является behavioral policy |
| NBV Head (PPO) | ❄️ Заморожен | В PPO не является политикой |
| ActorMLP (SAC) | 🔥 Обучается | Learned policy для SAC updates |
| QNetwork + Q_target (SAC) | 🔥 Обучается | Оценка Q-функции |
| log_std (SAC) | 🔥 Обучается | Стохастичность политики |
| log_alpha (SAC) | 🔥 Обучается | Auto entropy tuning |
| PPOActorCritic (PPO) | 🔥 Обучается | Вся политика и критик |

### Режим без флага (end-to-end)

Весь ODIN backbone также обучается через actor_optimizer (SAC) или ODIN не участвует в backprop PPO.

### Флаг `--train_last_transformer_block`

Дополнительно размораживает последний блок трансформера ODIN при `--freeze_backbone`. Компромисс между гибкостью и стабильностью обучения.

### Почему Coverage Head обучается отдельно от RL?

$p_{hidden}$ используется для вычисления $r_{coverage}$. Если бы градиент RL тёк через Coverage Head обратно в backbone:
1. Backbone начал бы деградировать (предсказывать $p_{hidden} = 0$ всегда → максимальный coverage reward).
2. Representation collapse: backbone оптимизировался бы не для распознавания, а для максимизации reward.

Поэтому: `p_hidden = sigmoid(logit).detach().item()` при вычислении reward — градиент не течёт.  
Coverage Head обучается только по BCE лоссу с GT-меткой из среды.

---

## 10. Конфигурация гиперпараметров

| Параметр | Значение | Смысл |
|---|---|---|
| `SCENE_STAGE` | 2 | Сцена: несколько объектов без препятствий (Stage 3 = с препятствиями — слишком сложно для старта) |
| `MIN_OBJECTS` | 2 | Минимум объектов в сцене |
| `MAX_OBJECTS` | 10 | Максимум объектов в сцене |
| `MAX_STEPS_PER_EPISODE` | 25 | 5 было слишком мало — агент не успевал исследовать сцену |
| `BUFFER_SIZE` | 10 000 | Ёмкость replay buffer (SAC) |
| `BATCH_SIZE` | 64 | Размер минибатча |
| `LEARNING_STARTS` | 500 | SAC начинает обновления только после 500 случайных шагов |
| `LEARNING_RATE` | $3 \times 10^{-4}$ | $1 \times 10^{-2}$ было слишком агрессивно для SAC |
| `REWARD_SCALE` | 20.0 | Множитель $\Delta p_{hidden}$ (само изменение мало — нужно масштабировать) |
| `PENALTY_COLLISION` | −15.0 | Строгий штраф за столкновение |
| `PENALTY_OOB` | −10.0 | Штраф за выход за рабочую зону |
| `REWARD_SURVIVAL` | 1.0 | Небольшой бонус за каждый безопасный шаг |
| $\gamma$ (discount) | 0.99 | Длинный горизонт планирования |
| $\tau$ (soft update) | 0.005 | Медленное обновление target network |
| $\varepsilon$ (PPO clip) | 0.2 | Стандартный PPO |
| $\lambda$ (GAE) | 0.95 | Стандартный PPO |

---

## Приложение: Структура файлов

```
train_odin_ppo_rl.py     ← PPO (дефолтный алгоритм)
train_odin_sac_rl.py     ← SAC (исправленная версия)
config.py                ← все гиперпараметры

src/simulation/
  environment_odin.py    ← NBVODINEnv (gym.Env)
  robot.py               ← Kuka IK
  camera.py              ← PyBullet RGB-D
  asset_loader.py        ← генерация сцены

src/vision/
  odin_adapter.py        ← ODINAdapter (обёртка ODIN)

scratch/my_odin/
  my_train_odin.py       ← NBVActiveODIN, CoverageHead, NBVHead
```
