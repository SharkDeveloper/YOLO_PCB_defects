#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для тестирования точности натренированной модели YOLO на тестовом наборе данных

Тестирование используется для окончательной оценки производительности модели на тестовом наборе данных.
Основные цели тестирования:
- Окончательная оценка точности модели с использованием метрик (precision, recall, mAP)
- Получение количественных показателей качества модели на невидимых данных
- Сравнение с результатами валидации для проверки переобучения

Отличие от валидации: 
- Тестирование выполняется на специально отведенном тестовом наборе данных
- Тестирование выполняется после завершения всего процесса обучения и валидации
- Тестирование дает окончательную оценку качества модели

Тестирование предоставляет метрики, которые показывают, насколько хорошо модель обобщает знания на новых данных.
"""

import sys
import os
import time
import torch
from ultralytics import YOLO
from utils import parse_yolo_output, log_validation_metrics, get_experiment_name

def check_cuda():
    """Check if CUDA is available and print GPU information"""
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        return True
    else:
        print("CUDA is not available. Testing will use CPU (slower).")
        print("To speed up testing, please install CUDA-compatible PyTorch version.")
        return False

def test_model(model_path=None, data_path="chip_defects.yaml", output_path="test_results"):
    """
    Тестирование точности натренированной модели YOLO на тестовом наборе данных
    
    Args:
        model_path (str): Путь к файлу модели (.pt)
        data_path (str): Путь к конфигурационному файлу данных (.yaml)
        output_path (str): Путь к директории для сохранения результатов
    """
    # Поиск модели, если путь не указан
    if model_path is None:
        model_paths = [
            "runs/detect/train_optimized/weights/best.pt",
            "runs/detect/train_optimized/weights/last.pt",
            "runs/detect/train5/weights/best.pt",
            "runs/detect/train5/weights/last.pt",
            "runs/detect/train/weights/best.pt",
            "runs/detect/train/weights/last.pt"
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            print("Модель не найдена. Пожалуйста, сначала натренируйте модель.")
            print("Ожидаемые пути к модели:")
            for path in model_paths:
                print(f"  - {path}")
            sys.exit(1)
    
    # Проверка существования файлов
    if not os.path.exists(model_path):
        print(f"Файл модели {model_path} не найден.")
        sys.exit(1)
        
    if not os.path.exists(data_path):
        print(f"Файл данных {data_path} не найден.")
        sys.exit(1)
    
    # Создание директории для результатов
    os.makedirs(output_path, exist_ok=True)
    
    # Check CUDA availability
    cuda_available = check_cuda()
    
    # Генерация имени эксперимента
    experiment_name = get_experiment_name("test")
    
    # Set validation parameters
    val_params = {
        'data': data_path,
        'split': 'test',  # Использовать тестовый набор данных
        'imgsz': 640,  # Размер изображения
        'batch': 16,  # Размер батча
        'conf': 0.25,  # Порог уверенности
        'iou': 0.45,  # Порог IoU для NMS
        'device': 0 if cuda_available else 'cpu',
        'project': output_path,
        'name': 'test',
        'exist_ok': True
    }
    
    print("Параметры тестирования:")
    for k, v in val_params.items():
        print(f"{k}: {v}")
    
    print(f"Тестирование модели: {model_path}")
    print(f"Результаты будут сохранены в: {output_path}")
    print("-" * 50)
    
    # Начало измерения времени
    start_time = time.time()
    
    try:
        # Load model
        model = YOLO(model_path)
        
        # Validate model on test set directly through API
        print("Запущено тестирование модели. Логи:")
        print("-" * 50)
        
        # Capture output for metrics parsing
        import io
        from contextlib import redirect_stdout
        
        # Create a string buffer to capture output
        output_buffer = io.StringIO()
        
        # Redirect stdout to capture YOLO logs
        with redirect_stdout(output_buffer):
            results = model.val(**val_params)
        
        # Get the captured output
        output_text = output_buffer.getvalue()
        output_lines = output_text.strip().split('\n')
        
        # Print captured output
        for line in output_lines:
            print(line)
        
        # Конец измерения времени
        end_time = time.time()
        
        print("\n" + "-" * 50)
        print("Тестирование модели завершено успешно!")
        print(f"Результаты сохранены в: {output_path}/test")
        
        # Парсинг метрик из вывода
        metrics = parse_yolo_output(output_lines)
        
        # Логирование метрик тестирования
        log_validation_metrics(
            experiment_name=experiment_name,
            start_time=start_time,
            end_time=end_time,
            metrics_data=metrics
        )
        
    except KeyboardInterrupt:
        print("\nТестирование прервано пользователем.")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при тестировании модели: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    """Основная функция"""
    # Активация виртуального окружения при необходимости
    venv_path = os.path.join(os.path.dirname(__file__), '..', 'venv')
    if os.path.exists(venv_path):
        site_packages = os.path.join(venv_path, 'Lib', 'site-packages')
        if site_packages not in sys.path:
            sys.path.insert(0, site_packages)
    
    # Парсинг аргументов командной строки
    import argparse
    parser = argparse.ArgumentParser(description="Тестирование точности натренированной модели YOLO на тестовом наборе данных")
    parser.add_argument("--model", type=str, help="Путь к файлу модели (.pt)")
    parser.add_argument("--data", type=str, default="chip_defects.yaml", help="Путь к конфигурационному файлу данных (.yaml)")
    parser.add_argument("--output", type=str, default="test_results", help="Путь к директории для сохранения результатов")
    
    args = parser.parse_args()
    
    test_model(model_path=args.model, data_path=args.data, output_path=args.output)

if __name__ == "__main__":
    main()

# Function for direct import and call
def run_test(output_path="test_results"):
    """
    Run model testing directly (for import and call from main.py)
    
    Args:
        output_path (str): Path to the output directory for test results
    """
    test_model(output_path=output_path)