# Ручная загрузка на Kaggle (из-за проблем с API)

## Проблема
Kaggle API не может подключиться из-за сетевых ограничений:
```
Connection aborted, ConnectionResetError(10054)
```

## Решение: Ручная загрузка через веб-интерфейс

### Шаг 1: Подготовка файлов
Все готово! Файлы находятся в:
```
notebooks/kaggle_training.ipynb
```

### Шаг 2: Загрузка на Kaggle

1. **Открой Kaggle**
   - Перейди на https://www.kaggle.com/code
   - Войди в аккаунт (sergeykurchev)

2. **Создай новый Notebook**
   - Нажми "New Notebook"
   - Выбери "Notebook" (не Script)

3. **Загрузи файл**
   - Нажми "File" → "Upload Notebook"
   - Выбери файл: `notebooks/kaggle_training.ipynb`
   - Подожди загрузки

4. **Настрой окружение**
   - В правой панели "Settings":
     - **Accelerator**: GPU T4 x2 (или P100)
     - **Internet**: ON (обязательно!)
     - **Persistence**: Files only

5. **Добавь датасет**
   - В правой панели "Input":
   - Нажми "+ Add Input"
   - Найди: `sergeykurchev/nbv-stage3-dataset`
   - Нажми "Add"

6. **Запусти обучение**
   - Нажми "Run All" (или Ctrl+Enter на каждой ячейке)
   - Обучение займёт ~4-7 часов

### Шаг 3: Мониторинг

Следи за прогрессом в notebook:
- ✅ Dataset download (~5 min)
- ✅ PointNet dataset generation (~10 min)
- ✅ PointNet training (~1-2 hours)
- ✅ RL training (~2-4 hours)
- ✅ Results packaging (~1 min)

### Шаг 4: Скачивание результатов

После завершения:
1. Перейди на вкладку "Output"
2. Найди файл `kaggle_results.zip`
3. Нажми "Download"

## Содержимое результатов

```
kaggle_results.zip:
├── weights/
│   ├── pointnet_best.pt          # Лучшая модель PointNet
│   ├── pointnet_last.pt           # Последний checkpoint
│   └── pointnet_metrics.csv       # Метрики обучения
├── rl_run/
│   ├── best_policy.zip            # Лучшая RL политика
│   ├── last_policy.zip            # Последний checkpoint
│   ├── rl_metrics.csv             # RL метрики
│   └── train_cnn_probs.csv        # CNN вероятности
└── SUMMARY.txt                    # Сводка обучения
```

## Альтернатива: Попробуй позже

Если хочешь использовать API:
```bash
cd notebooks
kaggle kernels push
```

Возможно, проблема временная и API заработает позже.

## Важные замечания

1. **Репозиторий обновлён**: Используется `https://github.com/SergKurchev/article-nbv.git`
2. **Stage 3 датасет**: Самый сложный (объекты + препятствия)
3. **Reduced timesteps**: RL обучение 50k шагов (вместо 500k) для Kaggle лимитов
4. **CSV логи**: Все метрики сохраняются в CSV для анализа

## Продолжение обучения локально

После скачивания `kaggle_results.zip`:

```bash
# Распаковать
unzip kaggle_results.zip

# Скопировать веса
cp -r kaggle_results/weights/* weights/primitives/
cp -r kaggle_results/rl_run runs/

# Продолжить обучение
uv run python train.py --load best
```

## Troubleshooting

### "Dataset not found"
- Убедись что добавил `sergeykurchev/nbv-stage3-dataset` в Input

### "Out of memory"
- Уменьши batch size в ячейке конфигурации
- Используй GPU T4 x2 (не CPU)

### "Timeout"
- Kaggle лимит 9-12 часов
- Частичные результаты всё равно сохранятся
- Можно продолжить локально

### "Repository not found"
- Убедись что репозиторий публичный: https://github.com/SergKurchev/article-nbv
- Или загрузи код как Kaggle dataset

---

**Готово к запуску!** Просто загрузи notebook через веб-интерфейс Kaggle.
