# Использование натренированной модели YOLO для детекции дефектов на PCB

## Запуск инференса

После тренировки модели вы можете использовать её для детекции дефектов на изображениях PCB.

### Базовое использование

Для запуска инференса с параметрами по умолчанию:

```bash
python src/infer.py
```

Или для использования в автоматизированном пайплайне:

```bash
python main.py --skip data_preparation training validation metrics visualization
```

## Подготовка данных

Для автоматической загрузки и подготовки датасета:

```bash
python src/data_preparation.py
```

Или для использования в автоматизированном пайплайне:

```bash
python main.py --skip training validation inference metrics visualization
```

По умолчанию скрипт:
- Использует последнюю натренированную модель (ищет в порядке приоритета)
- Обрабатывает изображения из `datasets/pcb/images/val`
- Сохраняет результаты в директорию `results`

### Указание конкретной модели

Если вы хотите использовать конкретную модель:

```bash
python src/infer.py --model runs/detect/train_optimized/weights/best.pt
```

### Указание собственных изображений

Для обработки собственных изображений:

```bash
python src/infer.py --source path/to/your/images
```

### Указание выходной директории

Для сохранения результатов в другую директорию:

```bash
python src/infer.py --output my_results
```

### Комбинирование параметров

Вы можете комбинировать параметры:

```bash
python src/infer.py --model runs/detect/train_optimized/weights/best.pt --source datasets/pcb/images/test --output test_results
```

## Тренировка модели

Для тренировки модели с параметрами по умолчанию:

```bash
python src/train_model.py
```

Или для использования в автоматизированном пайплайне:

```bash
python main.py --skip data_preparation validation inference metrics visualization
```

## Результаты инференса

После выполнения инференса в выходной директории будут созданы следующие файлы:

- `*.jpg` - изображения с наложенными bounding box'ами обнаруженных дефектов
- `labels/*.txt` - текстовые файлы с координатами bounding box'ов и уверенностью модели
- `predictions.json` - результаты в формате JSON (если включено)

## Использование модели в собственном коде

Вы также можете использовать натренированную модель напрямую в Python коде:

```python
from ultralytics import YOLO

# Загрузка модели
model = YOLO('runs/detect/train_optimized/weights/best.pt')

# Запуск инференса на изображении
results = model('path/to/image.jpg')

# Обработка результатов
for result in results:
    boxes = result.boxes  # Bounding box координаты
    masks = result.masks  # Маски сегментации (если есть)
    keypoints = result.keypoints  # Ключевые точки (если есть)
    probs = result.probs  # Вероятности классов (если есть)
    
    # Визуализация результатов
    result.show()  # Показать изображение с bounding box'ами
    result.save(filename='result.jpg')  # Сохранить изображение с bounding box'ами
```

## Проверка точности модели

Для оценки точности натренированной модели на тестовых данных используйте скрипт валидации:

```bash
python src/validate_model.py
```

Или для использования в автоматизированном пайплайне:

```bash
python main.py --skip data_preparation training inference metrics visualization
```

Этот скрипт автоматически найдет лучшую доступную модель и выполнит валидацию на тестовых данных, показывая метрики точности:
- mAP (mean Average Precision)
- Precision и Recall для каждого класса дефектов
- F1-score

Вы также можете указать конкретную модель и данные:

```bash
python src/validate_model.py --model runs/detect/train_optimized/weights/best.pt --data chip_defects.yaml
```

## Диагностика проблем

Если модель показывает нулевую точность или другие проблемы, используйте скрипт диагностики:

```bash
python src/diagnose_model.py
```

Или для использования в автоматизированном пайплайне:

```bash
python main.py --skip data_preparation training validation inference metrics visualization
```

Этот скрипт проверит:
- Конфигурационный файл данных
- Файл модели
- Образцы изображений и их разметку
- Логи тренировки

## Параметры командной строки

- `--model PATH` - путь к файлу модели (.pt)
- `--source PATH` - путь к изображениям для обработки
- `--output PATH` - путь к директории для сохранения результатов (по умолчанию: "results")

## Решение проблем

### Модель не найдена

Если вы получаете ошибку "No trained model found", убедитесь что:
1. Вы успешно завершили тренировку модели
2. Модель сохранена в одной из стандартных директорий:
   - `runs/detect/train_optimized/weights/best.pt`
   - `runs/detect/train5/weights/best.pt`
   - `runs/detect/train/weights/best.pt`

### Ошибка памяти

Если вы сталкиваетесь с ошибками памяти при обработке изображений:
1. Уменьшите размер батча в параметрах инференса
2. Используйте изображения меньшего размера
3. Закройте другие приложения, использующие GPU

### Низкая точность детекции

Если модель плохо детектирует дефекты:
1. Убедитесь, что изображения имеют хорошее качество
2. Попробуйте настроить порог уверенности (`conf` параметр)
3. Рассмотрите возможность дообучения модели на дополнительных данных

### Модель не обнаруживает дефекты

Если модель не обнаруживает дефекты на изображениях, где они должны быть:
1. Проверьте точность модели с помощью скрипта валидации
2. Убедитесь, что данные были правильно размечены
3. Попробуйте уменьшить порог уверенности (conf)
4. Рассмотрите возможность повторной тренировки с другими гиперпараметрами

### Нулевая точность модели (mAP = 0)

Если валидация показывает нулевую точность:
1. Выполните диагностику с помощью `src/diagnose_model.py`
2. Проверьте правильность разметки обучающих данных
3. Убедитесь, что в конфигурационном файле указаны правильные пути
4. Проверьте логи тренировки на наличие ошибок
5. Рассмотрите возможность повторной тренировки с другими параметрами

## Сравнение экспериментов

Для сравнения результатов разных экспериментов:

```bash
python src/compare_experiments.py
```

Или для использования в автоматизированном пайплайне:

```bash
python main.py --skip data_preparation training validation inference visualization
```

## Визуализация результатов

Для визуализации результатов тренировки и сравнения экспериментов:

```bash
python src/visualize.py
```

Или для использования в автоматизированном пайплайне:

```bash
python main.py --skip data_preparation training validation inference metrics
```

## Сравнение экспериментов

Для сравнения результатов разных экспериментов:

```bash
python src/compare_experiments.py
```

Или для использования в автоматизированном пайплайне:

```bash
python main.py --skip data_preparation training validation inference visualization
```