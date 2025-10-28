#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Основной скрипт для автоматизации всего процесса:
1. Подготовка данных
2. Тренировка модели
3. Валидация модели
4. Инференс
5. Сбор метрик
6. Визуализация

Все этапы выполняются последовательно с версионированием по времени.
"""

import os
import sys
import time
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Импортируем наши модули
from utils import parse_yolo_output, log_training_metrics, log_validation_metrics, log_inference_metrics, get_experiment_name
from visualize import visualize_training_metrics, visualize_experiment_comparison
from data_preparation import run_data_preparation as direct_run_data_preparation
from train_model import run_training as direct_run_training
from validate_model import run_validation as direct_run_validation
from test_model import run_test as direct_run_test
from infer import run_inference_direct as direct_run_inference
from compare_experiments import compare_experiments as direct_compare_experiments

def create_versioned_directory(base_path):
    """
    Создает директорию с версионированием по времени
    
    Args:
        base_path (str): Базовый путь
        
    Returns:
        str: Путь к версионированной директории
    """
    # Получаем текущее время для версионирования
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = f"{base_path}_{timestamp}"
    
    # Создаем директорию
    os.makedirs(versioned_path, exist_ok=True)
    
    return versioned_path

def run_data_preparation():
    """
    Запуск подготовки данных
    """
    print("=" * 60)
    print("1. Подготовка данных")
    print("=" * 60)
    
    try:
        # Создаем версионированную директорию для датасета
        dataset_base_path = "datasets/pcb"
        if not os.path.exists(dataset_base_path):
            os.makedirs(dataset_base_path, exist_ok=True)
        
        # Запуск подготовки данных напрямую
        direct_run_data_preparation()
        
        print("Подготовка данных завершена успешно!")
        return True
        
    except Exception as e:
        print(f"Ошибка при подготовке данных: {e}")
        return False

def run_training():
    """
    Запуск тренировки модели
    """
    print("=" * 60)
    print("2. Тренировка модели")
    print("=" * 60)
    
    try:
        # Создаем версионированную директорию для результатов тренировки
        train_base_path = "runs/detect/train"
        versioned_train_path = create_versioned_directory(train_base_path)
        
        # Запуск тренировки напрямую
        direct_run_training()
        
        print("Тренировка модели завершена успешно!")
        return True
        
    except Exception as e:
        print(f"Ошибка при тренировке модели: {e}")
        return False

def run_validation():
    """
    Запуск валидации модели
    """
    print("=" * 60)
    print("3. Валидация модели")
    print("=" * 60)
    
    try:
        # Создаем версионированную директорию для результатов валидации
        validation_base_path = "validation_results"
        versioned_validation_path = create_versioned_directory(validation_base_path)
        
        # Запуск валидации напрямую
        direct_run_validation(output_path=versioned_validation_path)
        
        print("Валидация модели завершена успешно!")
        return True
        
    except Exception as e:
        print(f"Ошибка при валидации модели: {e}")
        return False

def run_testing():
    """
    Запуск тестирования модели
    """
    print("=" * 60)
    print("4. Тестирование модели")
    print("=" * 60)
    
    try:
        # Создаем версионированную директорию для результатов тестирования
        test_base_path = "test_results"
        versioned_test_path = create_versioned_directory(test_base_path)
        
        # Запуск тестирования напрямую
        direct_run_test(output_path=versioned_test_path)
        
        print("Тестирование модели завершено успешно!")
        return True
        
    except Exception as e:
        print(f"Ошибка при тестировании модели: {e}")
        return False

def run_inference():
    """
    Запуск инференса
    """
    print("=" * 60)
    print("5. Инференс")
    print("=" * 60)
    
    try:
        # Создаем версионированную директорию для результатов инференса
        results_base_path = "results"
        versioned_results_path = create_versioned_directory(results_base_path)
        
        # Запуск инференса напрямую
        direct_run_inference(output_path=versioned_results_path)
        
        print("Инференс завершен успешно!")
        return True
        
    except Exception as e:
        print(f"Ошибка при инференсе: {e}")
        return False

def run_metrics_collection():
    """
    Сбор и отображение метрик
    """
    print("=" * 60)
    print("5. Сбор метрик")
    print("=" * 60)
    
    try:
        # Вызываем функцию для отображения метрик напрямую
        direct_compare_experiments()
        
        print("Сбор метрик завершен успешно!")
        return True
        
    except Exception as e:
        print(f"Ошибка при сборе метрик: {e}")
        return False

def run_visualization():
    """
    Запуск визуализации с помощью matplotlib
    """
    print("=" * 60)
    print("6. Визуализация")
    print("=" * 60)
    
    try:
        # Создаем директорию для отчетов
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        # Визуализация метрик обучения
        print("Генерация визуализации метрик обучения...")
        visualize_training_metrics()
        
        # Визуализация сравнения экспериментов
        print("Генерация визуализации сравнения экспериментов...")
        visualize_experiment_comparison()
        
        print("Визуализация завершена успешно!")
        return True
        
    except Exception as e:
        print(f"Ошибка при визуализации: {e}")
        return False

def run_complete_pipeline(skip_steps=None):
    """
    Запуск полного пайплайна
    
    Args:
        skip_steps (list): Список шагов для пропуска
    """
    if skip_steps is None:
        skip_steps = []
    
    print("Запуск полного пайплайна обработки PCB дефектов")
    print("=" * 60)
    start_time = time.time()
    
    # Шаг 1: Подготовка данных
    if "data_preparation" not in skip_steps:
        if not run_data_preparation():
            print("Ошибка на этапе подготовки данных. Прерывание пайплайна.")
            return False
    else:
        print("Пропущен этап подготовки данных")
    
    # Шаг 2: Тренировка модели
    if "training" not in skip_steps:
        if not run_training():
            print("Ошибка на этапе тренировки модели. Прерывание пайплайна.")
            return False
    else:
        print("Пропущен этап тренировки модели")
    
    # Шаг 3: Валидация модели
    if "validation" not in skip_steps:
        if not run_validation():
            print("Ошибка на этапе валидации модели. Прерывание пайплайна.")
            return False
    else:
        print("Пропущен этап валидации модели")
    
    # Шаг 4: Тестирование модели
    if "testing" not in skip_steps:
        if not run_testing():
            print("Ошибка на этапе тестирования модели. Прерывание пайплайна.")
            return False
    else:
        print("Пропущен этап тестирования модели")
    
    # Шаг 5: Инференс
    if "inference" not in skip_steps:
        if not run_inference():
            print("Ошибка на этапе инференса. Прерывание пайплайна.")
            return False
    else:
        print("Пропущен этап инференса")
    
    # Шаг 6: Сбор метрик
    if "metrics" not in skip_steps:
        if not run_metrics_collection():
            print("Ошибка на этапе сбора метрик. Прерывание пайплайна.")
            return False
    else:
        print("Пропущен этап сбора метрик")
    
    # Шаг 7: Визуализация
    if "visualization" not in skip_steps:
        if not run_visualization():
            print("Ошибка на этапе визуализации. Прерывание пайплайна.")
            return False
    else:
        print("Пропущен этап визуализации")
    
    # Общее время выполнения
    end_time = time.time()
    total_time = end_time - start_time
    
    print("=" * 60)
    print("Полный пайплайн выполнен успешно!")
    print(f"Общее время выполнения: {total_time:.2f} секунд")
    print("=" * 60)
    
    return True

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description="Полный пайплайн обработки PCB дефектов")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["data_preparation", "training", "validation", "testing", "inference", "metrics", "visualization"],
                        help="Этапы для пропуска")
    
    args = parser.parse_args()
    
    # Запуск полного пайплайна
    success = run_complete_pipeline(skip_steps=args.skip)
    
    if success:
        print("Все этапы пайплайна выполнены успешно!")
        sys.exit(0)
    else:
        print("Пайплайн завершен с ошибками!")
        sys.exit(1)

if __name__ == "__main__":
    main()