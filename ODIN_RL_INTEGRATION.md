# Интеграция NBVActiveODIN в Reinforcement Learning пайплайн

Этот документ описывает структуру и использование модифицированной среды обучения, которая использует **ODIN** (3D Scene Understanding) в качестве сенсорного движка для автономной навигации робота.

## 🚀 Архитектура: NBV Head = RL Agent

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
  NBV Head (Actor) ◄── pose       Reward = Δp_hidden
       │
       ▼
  next_pos (3D) + next_quat (4D)  ←─  ACTION
       │
       ▼
  PyBullet: робот перемещает камеру
```

**Ключевая идея:** NBV Head — это **и есть политика** RL-агента.
Отдельного SAC/PPO агента (SB3) **нет**. Вся политика живёт внутри ODIN.

## 🧠 Умная архитектура Q-Network (Twin Critics)

Вместо стандартного полносвязного MLP на сыром векторе состояния и абсолютном действии, реализована **геометрически-инвариантная, семантически-направленная Q-Network**:

1. **Относительные геометрические фичи (Geometric Relation)**:
   Входное действие (абсолютная целевая поза камеры) преобразуется в относительную систему координат текущего состояния:
   - `rel_pos = target_pos - current_pos` (относительное смещение).
   - `rel_euler = target_euler - current_euler` (относительные углы).
   - `dist = norm(rel_pos)` и `direction = rel_pos / dist` (расстояние и направление движения).
   - `hint_rel_pos = target_pos - nbv_hint` (отклонение от подсказки предобученной NBV Head).
2. **Семантика сцены (Scene Semantics)**:
   К 18-мерному вектору состояния прибавляется 256-мерный **Scene Embedding** (mean-pool из query-фичей декодера ODIN). Это даёт Q-сети полноценное представление о геометрии и объектах сцены. Внутри Q-сети этот эмбеддинг сжимается проекционным слоем до 64-мерного признакового пространства.
3. **Анализ безопасности (Workspace OOB Features)**:
   Вычисляются явные штрафные фичи выхода за пределы рабочей зоны (Out-Of-Bounds):
   - `oob_low = max(0, OOB_MIN - target_pos)`
   - `oob_high = max(0, target_pos - OOB_MAX)`
   Это позволяет критику мгновенно оценивать безопасность предложенного действия.

Благодаря этому критик обучается значительно стабильнее, обобщается на новые сцены и эффективнее штрафует за опасные движения робота.

## 📁 Структура проекта

| Файл | Описание |
|------|----------|
| `train_odin_sac_rl.py` | **SAC обучение** ⭐ — NBV Head = Actor, Twin Q, Replay Buffer |
| `train_odin_nbv_rl.py` | REINFORCE обучение — альтернатива для отладки |
| `train_odin_rl.py` | *(устаревший)* SAC с отдельным агентом SB3 |
| `src/vision/odin_adapter.py` | Адаптер ODIN: `infer()` и `infer_for_rl()` (с градиентами) |
| `src/simulation/environment_odin.py` | Gym-среда `NBVODINEnv` |
| `notebooks/kaggle_odin_rl_train.ipynb` | Kaggle-ноутбук для запуска SAC |
| `tests/test_odin_sac_agent.py` | Мок-тесты SAC (7 тестов) |
| `tests/test_odin_nbv_agent.py` | Мок-тесты REINFORCE (5 тестов) |

## 🔧 Три режима заморозки

| Флаги | Что обучается | Что заморожено | GPU | Скорость |
|------|---------------|----------------|-----|----------|
| `--freeze_backbone` | Только NBV Head | Backbone + Coverage Head | ~4 GB | Максимальная |
| `--freeze_backbone --train_last_transformer_block` | NBV Head + последний блок трансформера в декодере | Остальной Backbone + Coverage Head | ~6 GB | Высокая |
| *(без флага)* | Вся сеть (End-to-End) | Ничего | ~12+ GB | Низкая |

## 🎯 SAC vs REINFORCE

| | REINFORCE | SAC ⭐ |
|---|---|---|
| Тип | On-policy | **Off-policy** |
| Replay Buffer | Нет | **50k transitions** |
| Sample efficiency | Низкая | **Высокая** |
| Стабильность | Высокая дисперсия | **Twin Q + entropy** |

## 🚀 Запуск

### SAC (рекомендуется)
```bash
python train_odin_sac_rl.py \
    --odin_weights /path/to/model_final.pth \
    --odin_cfg my_odin/configs/scannet_context/3d.yaml \
    --freeze_backbone \
    --total_steps 100000
```

### REINFORCE (для отладки)
```bash
python train_odin_nbv_rl.py \
    --odin_weights /path/to/model_final.pth \
    --odin_cfg my_odin/configs/scannet_context/3d.yaml \
    --freeze_backbone \
    --total_episodes 500
```

## 🧪 Обучение на Kaggle

**Датасет сцен НЕ нужен** — PyBullet генерирует сцены из 3D-мешей в репозитории.

**Шаги запуска:**
1. Загрузите веса ODIN (`model_final.pth`) как Kaggle Dataset.
2. Откройте `notebooks/kaggle_odin_rl_train.ipynb`.
3. Подключите датасет с весами в Kaggle Inputs.
4. Включите GPU (T4 x2) и Internet: On.
5. Запустите ячейки 1 → 2 → 3 → 4.

## 🎥 Визуализация и Дашборд (Видео)
При включении записи видео (`--record_episodes`), скрипт сохраняет файлы, состоящие из трех частей:
1. **Camera View**: То, что видит робот (RGB).
2. **Stats Panel**:
    - `Step`: Текущий шаг в эпизоде.
    - `p_hidden`: Текущий уровень неопределенности (чем ниже, тем лучше изучена сцена).
    - `Ep.Rew`: Накопленная награда за эпизод.
3. **Live Plots**:
    - График `p_hidden`: Показывает, как быстро агент "раскрывает" скрытые зоны.
    - График `Reward`: Показывает мгновенную награду.

## 📊 Интерпретация метрик
- **Success Criteria**: Если `p_hidden` стабильно падает → агент научился находить ракурсы, "пробивающие" окклюзии.
- **alpha (SAC)**: Коэффициент энтропии. Высокий = больше исследования, низкий = эксплуатация.
- **critic_loss**: Ошибка Q-networks. Должна уменьшаться со временем.
- **Collisions**: Штраф за столкновение (`PENALTY_COLLISION = -15`) учит агента облетать препятствия.

## 🛠 Тестирование (Local)
```bash
# SAC тесты (без GPU/ODIN)
python -m pytest tests/test_odin_sac_agent.py -v

# REINFORCE тесты
python -m pytest tests/test_odin_nbv_agent.py -v

# Все тесты
python -m pytest tests/test_odin_sac_agent.py tests/test_odin_nbv_agent.py -v
```
