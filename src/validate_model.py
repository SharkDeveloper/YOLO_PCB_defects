#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для валидации точности натренированной модели YOLO

Валидация используется для оценки производительности модели на валидационном наборе данных.
Основные цели валидации:
- Оценка точности модели с использованием метрик (precision, recall, mAP)
- Проверка на переобучение
- Получение количественных показателей качества модели

Отличие от инференса: 
- Валидация фокусируется на оценке качества модели и вычислении метрик точности
- Инференс фокусируется на получении предсказаний для реальных задач обнаружения дефектов

Валидация обычно выполняется на специально отведенном валидационном наборе данных и 
предоставляет метрики, которые помогают понять, насколько хорошо модель обобщает знания.
"""

import subprocess
import sys
import os
import time
import argparse
from src.utils import parse_yolo_output, log_validation_metrics, get_experiment_name

def validate_model(model_path=None, data_path="chip_defects.yaml", output_path="validation_results"):
    """
    Валидация точности натренированной модели YOLO
    
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
    
    # Генерация имени эксперимента
    experiment_name = get_experiment_name("validation")
    
    # Проверка существования файлов
    if not os.path.exists(model_path):
        print(f"Файл модели {model_path} не найден.")
        sys.exit(1)
        
    if not os.path.exists(data_path):
        print(f"Файл данных {data_path} не найден.")
        sys.exit(1)
    
    # Создание директории для результатов
    os.makedirs(output_path, exist_ok=True)
    
    # Команда для валидации
    cmd = [
        'yolo',
        'val',  # Валидация
        'task=detect',  # Тип задачи
        f'model={model_path}',
        f'data={data_path}',
        'imgsz=640',  # Размер изображения
        'batch=16',  # Размер батча
        'conf=0.25',  # Порог уверенности
        'iou=0.45',  # Порог IoU для NMS
        f'project={output_path}',
        'name=validation',
        'exist_ok=True'
    ]
    
    print(f"Валидация модели: {model_path}")
    print(f"Данные: {data_path}")
    print(f"Результаты будут сохранены в: {output_path}")
    print("-" * 50)
    
    # Начало измерения времени
    start_time = time.time()
    
    # Сбор строк вывода для парсинга метрик
    output_lines = []
    
    # Выполнение валидации с потоковой передачей вывода
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace'
        )
        
        print("Запущена валидация модели. Логи:")
        print("-" * 50)
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                output_lines.append(output.strip())
                sys.stdout.flush()
        
        process.wait()
        
        # Конец измерения времени
        end_time = time.time()
        
        if process.returncode == 0:
            print("\n" + "-" * 50)
            print("Валидация модели завершена успешно!")
            print(f"Результаты сохранены в: {output_path}/validation")
            
            # Парсинг метрик из вывода
            metrics = parse_yolo_output(output_lines)
            
            # Логирование метрик валидации
            log_validation_metrics(
                experiment_name=experiment_name,
                start_time=start_time,
                end_time=end_time,
                metrics_data=metrics
            )
        else:
            print(f"\nВалидация завершена с ошибкой. Код возврата: {process.returncode}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nВалидация прервана пользователем.")
        process.terminate()
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при валидации модели: {e}")
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
    parser = argparse.ArgumentParser(description="Валидация точности натренированной модели YOLO")
    parser.add_argument("--model", type=str, help="Путь к файлу модели (.pt)")
    parser.add_argument("--data", type=str, default="chip_defects.yaml", help="Путь к конфигурационному файлу данных (.yaml)")
    parser.add_argument("--output", type=str, default="validation_results", help="Путь к директории для сохранения результатов")
    
    args = parser.parse_args()
    
    validate_model(model_path=args.model, data_path=args.data, output_path=args.output)

if __name__ == "__main__":
    main()

# Function for direct import and call
def run_validation(output_path="validation_results"):
    """
    Run model validation directly (for import and call from main.py)
    
    Args:
        output_path (str): Path to the output directory for validation results
    """
    validate_model(output_path=output_path)