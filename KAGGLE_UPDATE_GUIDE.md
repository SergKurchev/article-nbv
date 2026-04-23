# Kaggle Dataset Update Guide

## Автоматическое обновление всех датасетов

### Быстрый старт

```bash
# Обновить все 3 стадии одной командой
uv run python scripts/update_kaggle_datasets.py

# Обновить только одну стадию
uv run python scripts/update_kaggle_datasets.py --stage 2

# Проверить что будет загружено (без реальной загрузки)
uv run python scripts/update_kaggle_datasets.py --dry-run
```

## Возможности скрипта

### ✅ Автоматические проверки
- Проверяет наличие Kaggle CLI
- Проверяет существование датасетов на Kaggle
- Считает количество сэмплов в каждой стадии
- Запрашивает подтверждение перед загрузкой

### ✅ Умное обновление
- Автоматически создаёт новую версию если датасет существует
- Создаёт новый датасет если его ещё нет
- Добавляет timestamp и описание изменений
- Показывает ссылки на обновлённые датасеты

### ✅ Безопасность
- Dry-run режим для проверки
- Подтверждение перед загрузкой
- Детальный лог всех операций

## Параметры

```bash
--username USERNAME    # Kaggle username (default: sergeykurchev)
--key KEY             # Kaggle API key (default: из конфига)
--stage {1,2,3}       # Обновить только одну стадию
--notes "TEXT"        # Кастомное описание версии
--dry-run             # Показать что будет загружено без загрузки
--object-mode MODE    # primitives или complex (default: primitives)
```

## Примеры использования

### Обновить все датасеты с кастомным описанием
```bash
uv run python scripts/update_kaggle_datasets.py \
    --notes "Fixed multi-object generation and visualization"
```

### Обновить только Stage 3
```bash
uv run python scripts/update_kaggle_datasets.py --stage 3
```

### Проверить перед загрузкой
```bash
# Сначала dry-run
uv run python scripts/update_kaggle_datasets.py --dry-run

# Если всё ок, загрузить
uv run python scripts/update_kaggle_datasets.py
```

## Что делает скрипт

1. **Проверяет окружение**
   - Kaggle CLI установлен
   - Credentials настроены
   - Датасеты существуют локально

2. **Показывает информацию**
   ```
   Stage 1: 8 samples
   Stage 2: 8 samples
   Stage 3: 8 samples
   ```

3. **Запрашивает подтверждение**
   ```
   Ready to update 3 dataset(s) on Kaggle
   Continue? [y/N]:
   ```

4. **Загружает датасеты**
   - Создаёт ZIP архивы
   - Загружает на Kaggle
   - Создаёт новые версии

5. **Показывает результат**
   ```
   Stage 1: [OK] Success
     URL: https://www.kaggle.com/datasets/sergeykurchev/nbv-stage1-dataset
   Stage 2: [OK] Success
     URL: https://www.kaggle.com/datasets/sergeykurchev/nbv-stage2-dataset
   Stage 3: [OK] Success
     URL: https://www.kaggle.com/datasets/sergeykurchev/nbv-stage3-dataset
   ```

## Требования

```bash
# Установить Kaggle CLI
pip install kaggle

# Или через uv
uv pip install kaggle
```

## Troubleshooting

### Ошибка: "Kaggle CLI not found"
```bash
pip install kaggle
```

### Ошибка: "401 Unauthorized"
Проверь credentials в `~/.kaggle/kaggle.json`:
```json
{
  "username": "sergeykurchev",
  "key": "your-api-key"
}
```

### Ошибка: "Dataset not found"
Скрипт автоматически создаст новый датасет при первой загрузке.

## Ссылки на датасеты

- **Stage 1**: https://www.kaggle.com/datasets/sergeykurchev/nbv-stage1-dataset
- **Stage 2**: https://www.kaggle.com/datasets/sergeykurchev/nbv-stage2-dataset
- **Stage 3**: https://www.kaggle.com/datasets/sergeykurchev/nbv-stage3-dataset

---

**Версия**: 1.0  
**Дата**: 2026-04-23
