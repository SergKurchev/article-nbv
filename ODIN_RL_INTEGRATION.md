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

## 🔧 Два режима заморозки

| Флаг | Что обучается | Что заморожено | GPU | Скорость |
|------|---------------|----------------|-----|----------|
| `--freeze_backbone` | Только NBV Head | Backbone + Coverage Head | ~4 GB | Быстро |
| *(без флага)* | Вся сеть | Ничего | ~12+ GB | Медленно |

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
