# CLAUDE.md — Полная документация I/O и потока данных SAC-агента

Документ описывает **вход/выход SAC-агента**, **как информация течёт из среды → ODIN → агент → действие → среда**, и где именно берутся реварды. Файлы-источники указаны явно (file:line) для навигации.

---

## 1. Общая схема (high-level)

```
┌─────────────────────────────────────────────────────────────────┐
│  PyBullet Scene (Robot + objects + obstacles)                   │
│  ─ Robot (KUKA iiwa 7-DoF) держит «камеру» в end-effector       │
│  ─ Целевые объекты (1..N) + препятствия (Stage 3)               │
└───────────────┬─────────────────────────────────────────────────┘
                │ EE pose (pos, quat), joints, RGB-D, segmentation
                ▼
┌─────────────────────────────────────────────────────────────────┐
│  NBVODINEnv (Gymnasium)                                         │
│  ─ Собирает obs = {"image": RGBD(4,256,256), "vector": 18}      │
│  ─ Считает env-награды: collision/OOB/classifier/survival/...   │
└───────────────┬─────────────────────────────────────────────────┘
                │ obs (dict)
                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Train loop (train_odin_sac_rl.py)                              │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │ ODINAdapter.add_frame(rgb,depth,pose,K)                  │  │
│   │ ODINAdapter.infer_for_rl(cur_pos, cur_quat)              │  │
│   │  → p_hidden_logit, p_hidden, scene_emb(256), nbv_pos,    │  │
│   │    instances_3d, original_xyz                            │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│   obs_vec_full = obs["vector"](18, c nbv_hint в [15:18]) ⊕ scene_emb(256) │
│                                                                 │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │ ActorMLP(obs_vec_full) → mean(6) → Normal(mean, σ)       │  │
│   │ action_sample = rsample() ∈ R^6  (x,y,z,roll,pitch,yaw)  │  │
│   └──────────────────────────────────────────────────────────┘  │
└───────────────┬─────────────────────────────────────────────────┘
                │ action (np.float32, shape=(6,))
                ▼
┌─────────────────────────────────────────────────────────────────┐
│  env.step(action) → obs', rew_env, terminated, truncated, info  │
│  rew_full = rew_env + delta_p_hidden*scale + exploration_bonus  │
└─────────────────────────────────────────────────────────────────┘
                │ (obs_vec, action, rew_full, next_obs_vec, done)
                ▼
              ReplayBuffer → SAC updates (actor, twin-Q, α)
```

---

## 2. Архитектура нейросетей агента

Все три модуля определены в [train_odin_sac_rl.py](train_odin_sac_rl.py).

### 2.1 `ActorMLP` ([train_odin_sac_rl.py:105-130](train_odin_sac_rl.py#L105-L130))
| | |
|---|---|
| **Вход** | `obs ∈ R^{274}` = `vector(18) ⊕ scene_emb(256)` |
| **Архитектура** | `Linear(274,256) → LayerNorm → ReLU → Linear(256,256) → LayerNorm → ReLU → Linear(256,6)` |
| **Выход** | `mean ∈ R^6` — детерминированное среднее политики |
| **Стохастичность** | `std = exp(log_std).clamp(1e-4, 2.0)`, где `log_std` — обучаемый параметр shape `(6,)` ([train_odin_sac_rl.py:248](train_odin_sac_rl.py#L248)) |
| **Семплирование** | `Normal(mean, std).rsample()` (репараметризация для backprop через actor loss) |

`ActorMLP` является **И поведенческой**, И **обучаемой** политикой — это ключевой фикс над предыдущей версией (см. memory `architecture_decisions.md`).

### 2.2 `QNetwork` (Twin Q-critic) ([train_odin_sac_rl.py:137-205](train_odin_sac_rl.py#L137-L205))
Не принимает `(obs, action)` напрямую, а строит **гео-семантические признаки**:

| Признак | Формула | Размер |
|---|---|---|
| `scene_feats` | `scene_proj(obs[18:274])` (Linear 256→128→64) | 64 |
| `rel_pos` | `action[:3] - obs[:3]` | 3 |
| `rel_euler` | `action[3:] - obs[3:6] * 0.1` | 3 |
| `dist`, `direction` | `‖rel_pos‖`, `rel_pos / dist` | 1 + 3 |
| `hint_rel_pos`, `hint_dist` | `action[:3] - obs[15:18]` (nbv_hint) | 3 + 1 |
| `oob_feats` | `ReLU(min−tgt)`, `ReLU(tgt−max)` (penalty proxy) | 6 |
| `delta_p` | `obs[7:8]` | 1 |
| `joints` | `obs[8:15]` | 7 |
| **Итого** | concat → MLP(92→256→256→1) | **92** |

Twin-сеть: два независимых `(scene_proj_i, q_i_mlp)` для Q₁, Q₂.

### 2.3 `OdinSACAgent` ([train_odin_sac_rl.py:212-396](train_odin_sac_rl.py#L212-L396))
Содержит:
- `actor_mlp: ActorMLP`
- `log_std: nn.Parameter(6)` — глобальный (state-independent) `log σ`
- `critic: QNetwork`, `critic_target: deepcopy(critic)` — soft-update с `τ=0.005`
- `log_alpha` — обучаемая температура SAC (`auto_entropy=True`, `target_entropy=-6`)
- `odin_model` — NBVActiveODIN (ВЕСЫ ODIN ТРЕНИРУЮТСЯ как часть actor_params, если `requires_grad=True`!)

`actor_params = ActorMLP.parameters() + [log_std] + (ODIN NBV-head параметры, если не заморожены)` ([train_odin_sac_rl.py:254-260](train_odin_sac_rl.py#L254-L260)) — то есть один оптимизатор обновляет и MLP, и NBV-голову ODIN.

---

## 3. Вход агента — наблюдение `obs`

### 3.1 Состав `obs` из среды ([environment_odin.py:595-663](src/simulation/environment_odin.py#L595-L663))

`env.step()`/`env.reset()` возвращают `dict`:

```python
obs = {
    "image":  np.float32 shape=(4, 256, 256),  # RGB(3) ⊕ depth_normalized(1)
    "vector": np.float32 shape=(18,),
}
```

`obs["image"]` — нормализованный RGB-D (RGB/255, depth/10) в формате CHW.

**`obs["vector"]` (18-D) — структура** ([environment_odin.py:654-661](src/simulation/environment_odin.py#L654-L661)):

| Индекс | Поле | Источник | Описание |
|---|---|---|---|
| `[0:3]` | `pos` | `Robot.get_ee_pose()` (или `no_arm_pos`) | xyz end-effector'а в мировых координатах |
| `[3:7]` | `orn` | то же | ориентация EE как кватернион (x,y,z,w) |
| `[7:8]` | `delta_p_hidden` | `last_p_hidden - new_p_hidden` (от ODIN) | изменение «вероятности скрытых объектов» |
| `[8:15]` | `joints` | `Robot.get_joint_states()` | углы 7 суставов KUKA |
| `[15:18]` | `nbv_hint` | `ODINAdapter.last_nbv_pos` | подсказка NBV-головы ODIN (xyz следующей позиции) |

**ВАЖНО**: в SAC-трейнере поле `[15:18]` **перезаписывается** свежим `nbv_pos_t` из `infer_for_rl()` ПЕРЕД склейкой с `scene_emb` ([train_odin_sac_rl.py:300-303](train_odin_sac_rl.py#L300-L303)).

### 3.2 Откуда берутся RGB / Depth / Segmentation

Через [src/simulation/camera.py](src/simulation/camera.py):
- `Camera.get_image(cam_pos, cam_orn)` → PyBullet `getCameraImage` с `viewMatrix`, построенной из позы EE.
- Камера «смотрит» вдоль `forward = R[:,2]` (локальная +Z EE); `up = R[:,0]` ([camera.py:40-45](src/simulation/camera.py#L40-L45)).
- `eye_pos = cam_pos + forward * 0.05` (смещение от фланца, чтобы не видеть внутренности захвата).
- Depth линеаризуется из `[0,1]` буфера в метры ([camera.py:77](src/simulation/camera.py#L77)).
- `IMAGE_SIZE = 256`, `fov = 60°`, near/far = 0.1/10 m ([config.py:95-99](config.py#L95-L99)).

### 3.3 Финальный вход в ActorMLP

В рантайме ([train_odin_sac_rl.py:299-306](train_odin_sac_rl.py#L299-L306)):

```python
obs_vec        = obs["vector"].copy()                          # (18,)
obs_vec[15:18] = result["nbv_pos_t"].detach().cpu().numpy()    # перезатираем nbv_hint
scene_emb      = result["scene_embedding"]                     # (256,) от ODIN query_features
obs_vec_full   = np.concatenate([obs_vec, scene_emb])          # (274,)
```

`obs_vec_full` подаётся в ActorMLP. Он же сохраняется в replay buffer (см. §6).

---

## 4. Поток данных через ODIN

Все RGB-D кадры проходят через `ODINAdapter` ([src/vision/odin_adapter.py](src/vision/odin_adapter.py)).

### 4.1 Накопление кадрового буфера

`_add_obs_frame(obs, adapter)` ([train_odin_sac_rl.py:741-755](train_odin_sac_rl.py#L741-L755)) вызывает `adapter.add_frame(...)`:

```python
adapter.add_frame(
    rgb,                                  # (256,256,3) uint8 — обычное изображение
    depth = obs["image"][3] * 10.0,       # (256,256) float32, метры (умножаем обратно)
    pose  = 4x4 camera-to-world матрица из (cam_pos, cam_quat),
    fx=fy=cx=cy=128.0,                    # упрощённые интринзики (IMAGE_SIZE/2)
)
```

Внутри ([odin_adapter.py:131-177](src/vision/odin_adapter.py#L131-L177)):
- RGB → torch `(3,H,W) float32`
- Depth → torch `(H,W) float32`
- Поза → flip Z (OpenGL → ODIN-конвенция, +Z вперёд)
- Кадры хранятся в `_frame_buffer` (deque, max **5** кадров — `num_frames=5`).

### 4.2 `infer_for_rl()` — основной forward ODIN с градиентами

[odin_adapter.py:286-393](src/vision/odin_adapter.py#L286-L393). НЕ обёрнут в `no_grad()` — позволяет SAC-loss затекать в NBV-голову.

**Вход**:
- `_frame_buffer` (до 5 кадров RGB-D + поз)
- `current_pos: (3,)`, `current_quat: (4,)` — текущая поза EE

**Шаги**:
1. **Backprojection** (без градиентов): RGB-D + intrinsics + pose → 3D point cloud `original_xyz` через `odin.modeling.backproject.backprojector_dataloader` ([odin_adapter.py:409-436](src/vision/odin_adapter.py#L409-L436)).
2. **`_build_odin_input`** ([odin_adapter.py:438-496](src/vision/odin_adapter.py#L438-L496)): formирует detectron2-batched dict со всеми полями (`images, depths, poses, intrinsics, multi_scale_xyz, instances_all, coverage_gt, current_camera_position, current_camera_quaternion, rl_mode=True`, dummy `Instances`).
3. `outputs = self.model(batched_inputs)` — прямой проход через `NBVActiveODIN`.
4. Извлечение результата.

**Выход** (`dict`):

| Ключ | Тип | Откуда | Использование |
|---|---|---|---|
| `nbv_pos_t` | `Tensor (3,) с grad` | NBV-head | пишется в `obs_vec[15:18]` |
| `nbv_quat_t` | `Tensor (4,) с grad` | NBV-head | (опционально, не используется в SAC) |
| `p_hidden` | `float` | `sigmoid(coverage_head_logit).item()` | для расчёта `delta_p_hidden` |
| `p_hidden_logit_t` | `Tensor (1,) с grad` | Coverage Head (raw logit) | BCE-supervision (см. §5.4) |
| `delta_p_hidden` | `float` | `prev_p_hidden - p_hidden` | (не используется напрямую — пересчитывается в loop) |
| `instances_3d` | `dict / Instances` | сегментационная голова ODIN | передаётся в `env.step(info_from_adapter=...)` |
| `original_xyz` | `np.ndarray (N,3)` | бэкпроекция | для сопоставления instances ↔ GT objects |
| `scene_embedding` | `np.ndarray (256,)` | `query_features.mean(dim=1)` из `_last_outputs` | конкатенируется к `obs_vec` |

### 4.3 Scene embedding (256-D)

Берётся из внутреннего `_last_outputs["query_features"]` базовой ODIN-модели ([odin_adapter.py:368-376](src/vision/odin_adapter.py#L368-L376)):
```python
q_feats = last_outputs["query_features"]   # (B, Q, D=256)
scene_emb = q_feats.mean(dim=1).detach().cpu().numpy().squeeze(0)  # (256,)
```
Это **усреднение по query-токенам** — глобальное представление сцены, которое подаётся в ActorMLP и QNetwork.

---

## 5. Реварды — откуда что приходит

Финальный `reward` = `env_reward + reward_coverage + reward_exploration` ([train_odin_sac_rl.py:643-651](train_odin_sac_rl.py#L643-L651)).

### 5.1 `env_reward` — приходит из `env.step()` ([environment_odin.py:338-376](src/simulation/environment_odin.py#L338-L376))

| Условие | Значение `reward` | Источник конфига |
|---|---|---|
| Столкновение робота с препятствием/целью | `PENALTY_COLLISION = -5.0` + `terminated=True` | [config.py:126](config.py#L126) |
| Out-of-bounds (EE.z<0.05 или ‖xy‖>1.0) | `PENALTY_OOB = -5.0` | [config.py:122](config.py#L122) |
| Иначе | `rew_classifier + rew_all_found + rew_success_classified + rew_survival` | см. ниже |

- `rew_classifier = (mean_class_conf - last_class_conf) * REWARD_CLASSIFIER_SCALE (=5.0)` — рост уверенности классификатора ODIN (среднее `instances_3d.pred_scores`).
- `rew_all_found = +15.0` (`REWARD_ALL_FOUND_BONUS`) — разово, когда видели всех target'ов (по сегментации).
- `rew_success_classified = +20.0` + `terminated=True` + `success=True` — все target'ы правильно классифицированы.
- `rew_survival = +3.0` за каждый «нормальный» шаг.

### 5.2 OOB-инжекция в training loop

OOB детектится по **некипнутому** action ([train_odin_sac_rl.py:587-592](train_odin_sac_rl.py#L587-L592)):
```python
is_manual_oob = any(action_np[:3] < ACTION_MIN[:3]) or any(action_np[:3] > ACTION_MAX[:3])
action_np = np.clip(action_np, ACTION_MIN, ACTION_MAX)   # env получает clamp'нутое
```
Если `is_manual_oob and not is_collision`, цикл **переписывает** `_env_rew = PENALTY_OOB` (env не знал, что было OOB, т.к. видел clipped action).

### 5.3 `rew_coverage` — считается во внешнем loop, НЕ в env

[train_odin_sac_rl.py:601-617](train_odin_sac_rl.py#L601-L617):
```python
# После env.step добавляем следующий кадр в ODIN
_add_obs_frame(next_obs, adapter)

# Повторный инференс ODIN с новой позой → next_p_hidden
next_outputs = adapter.infer_for_rl(next_pos, next_quat)
next_p_hidden = float(next_outputs["p_hidden"])

delta_p_hidden = p_hidden - next_p_hidden     # >0 если ODIN стал увереннее
rew_coverage   = delta_p_hidden * REWARD_SCALE  # REWARD_SCALE = 10.0
```
Этот компонент **добавляется только если НЕ collision и НЕ OOB** ([train_odin_sac_rl.py:645-646](train_odin_sac_rl.py#L645-L646)).

### 5.4 Supervised обучение Coverage Head (отдельный loss, НЕ reward!)

[train_odin_sac_rl.py:619-628](train_odin_sac_rl.py#L619-L628). Параллельно с RL цикл тренирует coverage head классическим BCE:
```python
gt = 1.0 if env.num_hidden_objects > 0 else 0.0    # бинарная GT
loss_cov = BCE_with_logits(p_hidden_logit_t, gt)
coverage_optimizer.step()   # отдельный Adam для coverage_head params
```
**Это decoupled supervision** — не часть SAC-loss'а.

### 5.5 `rew_exploration` ([train_odin_sac_rl.py:646-651](train_odin_sac_rl.py#L646-L651))

```python
if p_hidden < REWARD_EXPLORATION_THRESHOLD (=0.05):
    rew_expl = REWARD_EXPLORATION_COMPLETE_BONUS (=5.0)
else:
    rew_expl = (1.0 - p_hidden) * REWARD_EXPLORATION_PARTIAL_FACTOR (=3.0)
```

### 5.6 Нормализация ревардов в буфере

[ReplayBuffer.sample()](train_odin_sac_rl.py#L81-L95): онлайн Welford mean/var, при выборке батча реварды нормируются `(r-μ)/(σ+ε)` — Q-функция работает в нормированном пространстве. Target-Q клипуется в `[-100, 100]` ([train_odin_sac_rl.py:354](train_odin_sac_rl.py#L354)).

---

## 6. Выход агента — действие, передача в среду

### 6.1 Формат action

`action_np ∈ R^6`, тип `np.float32`. Семантика — **абсолютная цель** в мировых координатах:
- `action_np[0:3] = [x, y, z]` — желаемая позиция камеры/EE
- `action_np[3:6] = [roll, pitch, yaw]` — желаемые углы Эйлера

Границы из конфига ([config.py:110-111](config.py#L110-L111)):
```python
ACTION_MIN = [0.2, -0.5, 0.0, -π, -π, -π]
ACTION_MAX = [0.8,  0.5, 0.8,  π,  π,  π]
```

ActorMLP выдаёт **ненормированный** R^6; **никакой `tanh`** на выходе нет. Клиппинг применяется ВНЕ актора, чтобы OOB-детектор успел увидеть «истинный» action и навесить `PENALTY_OOB`.

### 6.2 `env.step(action)` — что делает среда ([environment_odin.py:229-244](src/simulation/environment_odin.py#L229-L244))

```python
target_pos   = action[:3]
target_euler = action[3:]
target_orn   = p.getQuaternionFromEuler(target_euler)
self.robot.apply_action(target_pos, target_orn)
```

`Robot.apply_action` ([robot.py:41-68](src/simulation/robot.py#L41-L68)):
1. **Inverse kinematics**: `p.calculateInverseKinematics(robot_id, ee_index, target_pos, target_orn)` → 7 углов суставов.
2. **Teleport + motor control**: для каждого сустава вызывается `resetJointState` (мгновенное перемещение) И `setJointMotorControl2(POSITION_CONTROL, force=1000)` (физический контроль). Тельепортация нужна, чтобы избавиться от инерции/лага IK.
3. `p.stepSimulation()` — один шаг физики.

После этого:
- Среда проверяет коллизии (`p.getClosestPoints` с препятствиями и target-объектами при `not no_arm`).
- Зовёт `_get_obs()` → получает новые `(pos, orn, joints, RGB-D, seg)`.
- **`_get_obs` в режиме `external_infer=True`** (как в SAC) НЕ вызывает ODIN сама — она читает `adapter.last_p_hidden` и `adapter.last_nbv_pos`, оставшиеся от внешнего инференса ([environment_odin.py:641-649](src/simulation/environment_odin.py#L641-L649)).
- Считает env-награды (§5.1).
- Возвращает `(obs, reward, terminated, truncated, info)`. `info` содержит `delta_p_hidden`, `p_hidden`, `success`, `correct_objects`, `nbv_pos`.

### 6.3 Терминирование эпизода

- `terminated = True` при коллизии или `success` (все target правильно классифицированы).
- `truncated = True` при `step_count >= MAX_STEPS_PER_EPISODE (=10)` ([config.py:92](config.py#L92)).
- В цикле SAC: `done = terminated or truncated`. **OOB сам по себе не терминирует на стороне env**, но трейнер дополнительно вешает `PENALTY_OOB` и продолжает (есть исторические комментарии о том, что OOB должно терминировать — см. `config.py:121`).

---

## 7. SAC updates — что подаётся в обучение

### 7.1 Push в replay buffer ([train_odin_sac_rl.py:668](train_odin_sac_rl.py#L668))

```python
replay_buffer.push(
    obs_vec       = obs_vec_full,             # (274,) — с nbv_hint и scene_emb
    action_np     = action_np,                # (6,) — clipped
    reward        = float(reward),            # rew_env + rew_coverage + rew_expl
    next_obs_vec  = concat(next_obs["vector"], next_scene_emb),  # (274,)
    done          = float(terminated or truncated),
)
```

### 7.2 Гиперпараметры ([config.py:101-107](config.py#L101-L107))

| Параметр | Значение |
|---|---|
| `LEARNING_RATE` (default `--lr_critic`) | `3e-4` |
| `--lr_actor` | `1e-4` |
| `gamma` | `0.99` |
| `tau` (soft-update) | `0.005` |
| `BATCH_SIZE` | `64` |
| `BUFFER_SIZE` | `10000` |
| `LEARNING_STARTS` | `1000` |
| `GRADIENT_STEPS` (UTD ratio) | `4` |
| `init_alpha`, auto-entropy | `0.2`, target = `-action_dim = -6` |
| `grad_clip` | `1.0` |

### 7.3 Обновления

[train_odin_sac_rl.py:681-686](train_odin_sac_rl.py#L681-L686):
```python
for _ in range(args.gradient_steps):       # 4
    batch = replay_buffer.sample(64)
    c_loss = agent.update_critic(batch)
    a_loss, _ = agent.update_actor_and_alpha(batch)
agent.soft_update_target()
```

- **Critic update** ([train_odin_sac_rl.py:344-366](train_odin_sac_rl.py#L344-L366)): берёт `(s, a, r, s', d)`, считает `target_q = r + γ(1-d)(min(Q'_1, Q'_2)(s', a') - α·log π(a'|s'))`, минимизирует **Huber-loss** обоих Q (Huber вместо MSE для стабильности).
- **Actor update** ([train_odin_sac_rl.py:368-391](train_odin_sac_rl.py#L368-L391)): `actor_loss = E[α·log π(a|s) - min(Q_1, Q_2)(s, a)]`. `actor_optimizer` обновляет `ActorMLP + log_std + (NBV-head ODIN, если не заморожена)`.
- **α update**: `α_loss = -log α · (log π + target_entropy).detach()`.

---

## 8. Замораживание ODIN и градиенты

CLI флаги [train_odin_sac_rl.py:408-411](train_odin_sac_rl.py#L408-L411):
- `--freeze_backbone` — обучается только `nbv_head` + `coverage_head` ODIN (плюс опционально `--train_last_transformer_block`).
- Без флага — всё ODIN тренируется end-to-end.

Параметры `coverage_head` НЕ попадают в `actor_optimizer` — у них собственный `coverage_optimizer` ([train_odin_sac_rl.py:520-524](train_odin_sac_rl.py#L520-L524)). Так Coverage Head обучается ИСКЛЮЧИТЕЛЬНО supervised BCE сигналом, не RL-loss'ом.

Параметры `nbv_head` входят в `actor_params` ([train_odin_sac_rl.py:255-260](train_odin_sac_rl.py#L255-L260)) — значит SAC actor-loss теоретически течёт в NBV-голову через `obs_vec[15:18] = nbv_pos_t`. Но `nbv_pos_t` в Actor подаётся через **np-конвертацию** (`.detach().cpu().numpy()` в [train_odin_sac_rl.py:301](train_odin_sac_rl.py#L301)) — поэтому **на практике градиент в NBV-голову через actor НЕ течёт**. Если нужен grad-flow, эту цепочку надо переделать на тензорный путь.

---

## 9. Что выходит на каждом шаге (наглядный дамп)

Один шаг эпизода производит следующие тензоры/массивы (имена соответствуют переменным в `main()`):

| Имя | Shape | dtype | Что это |
|---|---|---|---|
| `obs["image"]` | `(4,256,256)` | float32 | RGB-D (нормализован) |
| `obs["vector"]` | `(18,)` | float32 | EE pose + delta_p + joints + nbv_hint |
| `result["nbv_pos_t"]` | `(3,)` | float32 tensor (grad) | подсказка NBV-head |
| `result["p_hidden"]` | scalar | float | вероятность скрытых объектов |
| `result["scene_embedding"]` | `(256,)` | float32 | усреднённые query features ODIN |
| `obs_vec_full` | `(274,)` | float32 | вход в ActorMLP |
| `action_np` (raw) | `(6,)` | float32 | выход ActorMLP + шум |
| `action_np` (clipped) | `(6,)` | float32 | то, что получает env |
| `_env_rew` | scalar | float | реward от env |
| `delta_p_hidden`, `rew_coverage`, `rew_expl` | scalar | float | компоненты пользовательской награды |
| `reward` | scalar | float | сумма; пишется в буфер |
| `next_obs_vec` | `(274,)` | float32 | вход в Q'(s', a') |
| `done` | scalar | float (0/1) | флаг конца эпизода |

---

## 10. Запуск

```bash
python train_odin_sac_rl.py \
    --odin_weights ./model_final.pth \
    --odin_cfg my_odin/configs/scannet_context/3d.yaml \
    --num_classes 24 \
    --freeze_backbone \
    --total_steps 200000 \
    --scene_stage 2 \
    --output_dir ./output_odin_sac
```

Артефакты в `output_dir`:
- `sac_metrics.csv` — пошаговая статистика (reward components, losses, α, collision/oob rate).
- `logs/training_metrics.csv` — пошаговые метрики от env.
- `videos/episode_<N>.gif` — для эпизодов из `--record_episodes`.
- `best.pth`, `ckpt_step<N>.pth`, `last.pth` — чекпоинты (`actor_mlp`, `critic`, `log_std`, `log_alpha`, ODIN heads).

---

## 11. Известные тонкости / ловушки

1. **Камеру тянет за EE робота**. «Камера» — это не отдельное тело: `Camera.get_image` строит view matrix из `(EE.pos, EE.orn)`, поэтому action управляет именно EE, а не камерой напрямую. Forward = локальная +Z EE ([camera.py:40](src/simulation/camera.py#L40)).
2. **`external_infer=True`** в SAC: env НЕ вызывает ODIN внутри `_get_obs()`. ODIN живёт во внешнем цикле; env лишь читает закешированные `adapter.last_p_hidden` / `last_nbv_pos`. Это позволяет одному forward'у ODIN покрыть и observation, и reward, и actor-input.
3. **`nbv_pos_t` в obs_vec — это input feature, не действие**. ActorMLP сам решает, насколько следовать подсказке NBV.
4. **`log_std` глобальный** — не зависит от состояния. Это упрощение по сравнению с типичным SAC (state-dependent σ через ещё один head).
5. **Reward normalization** в буфере — критическая стабилизация: без неё Q взрывается при больших `+15/+20` бонусах.
6. **OOB и collision не складываются**: при collision PENALTY_COLLISION уже выставлен, OOB не пересиливает (`is_oob = is_manual_oob and not is_collision`).
7. **Frame buffer ODIN на 5 кадров** — `infer_for_rl` всегда работает с до-5 последними observation; буфер чистится по `adapter.reset()` на каждом `env.reset()`.
