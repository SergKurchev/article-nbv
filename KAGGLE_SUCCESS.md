# ✅ Kernel успешно загружен на Kaggle!

## Статус

**Kernel создан**: https://www.kaggle.com/code/sergeykurchev/nbv-training-full-pipeline

**Версия**: 1  
**Дата**: 2026-04-23  
**Статус**: Готов к запуску

## ⚠️ Важно: GPU квота исчерпана

Kaggle показывает: "Maximum weekly GPU quota of 30.00 hours reached"

**Решение**: 
1. Kernel загружен с CPU (enable_gpu: false)
2. Когда квота восстановится, включи GPU вручную в настройках kernel
3. Квота обновляется еженедельно

## Как запустить обучение

### Вариант 1: Подождать восстановления GPU квоты

1. Открой kernel: https://www.kaggle.com/code/sergeykurchev/nbv-training-full-pipeline
2. Подожди восстановления квоты (проверяй в Settings → Accelerator)
3. Включи GPU: Settings → Accelerator → GPU T4 x2
4. Нажми "Run All"

### Вариант 2: Запустить на CPU (медленно, не рекомендуется)

1. Открой kernel: https://www.kaggle.com/code/sergeykurchev/nbv-training-full-pipeline
2. Нажми "Run All"
3. ⚠️ Обучение займёт 20-30 часов на CPU (вместо 4-7 на GPU)

### Вариант 3: Использовать другой аккаунт Kaggle

1. Создай новый Kaggle аккаунт (или используй другой)
2. Загрузи notebook вручную (файл: `notebooks/kaggle_training.ipynb`)
3. Добавь датасет: `sergeykurchev/nbv-stage3-dataset`
4. Включи GPU и запусти

## Что будет происходить при запуске

1. **Setup** (~2 min)
   - Установка uv
   - Клонирование репозитория: https://github.com/SergKurchev/article-nbv.git

2. **Dataset Download** (~5 min)
   - Загрузка Stage 3 датасета
   - ~8000 samples с объектами и препятствиями

3. **PointNet Dataset Generation** (~10 min)
   - Конвертация RGB-D в point clouds
   - Создание dataset_pointnet/

4. **PointNet Training** (~1-2 hours на GPU)
   - 50 epochs
   - Batch size: 32
   - CSV логи: pointnet_metrics.csv

5. **RL Training** (~2-4 hours на GPU)
   - 50,000 timesteps (reduced для Kaggle)
   - Headless mode (no GUI)
   - CSV логи: rl_metrics.csv

6. **Results Packaging** (~1 min)
   - Создание kaggle_results.zip
   - Готово к скачиванию

## Результаты

После завершения скачай `kaggle_results.zip` из Output tab:

```
kaggle_results/
├── weights/
│   ├── pointnet_best.pt
│   ├── pointnet_last.pt
│   └── pointnet_metrics.csv
├── rl_run/
│   ├── best_policy.zip
│   ├── last_policy.zip
│   ├── rl_metrics.csv
│   └── train_cnn_probs.csv
└── SUMMARY.txt
```

## Мониторинг

Следи за прогрессом в notebook:
- Каждая ячейка показывает прогресс
- CSV файлы обновляются в реальном времени
- Можно скачать промежуточные результаты

## Troubleshooting

### "GPU quota exceeded"
- Подожди восстановления квоты (еженедельно)
- Или используй другой аккаунт

### "Repository not found"
- Убедись что репозиторий публичный: https://github.com/SergKurchev/article-nbv
- Проверь что Internet включен в Settings

### "Dataset not found"
- Проверь что датасет добавлен в Input
- Slug: `sergeykurchev/nbv-stage3-dataset`

### "Out of memory"
- Уменьши batch size в ячейке конфигурации
- Используй GPU T4 x2 (не CPU)

## Продолжение локально

После скачивания результатов:

```bash
# Распаковать
unzip kaggle_results.zip

# Скопировать веса
cp -r kaggle_results/weights/* weights/primitives/
cp -r kaggle_results/rl_run runs/

# Продолжить обучение
uv run python train.py --load best
```

---

**Kernel готов к запуску!** Просто включи GPU когда квота восстановится. 🚀
