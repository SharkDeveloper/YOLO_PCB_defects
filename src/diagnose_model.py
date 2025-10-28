#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для диагностики проблем с моделью и данными
"""

import subprocess
import sys
import os
import argparse
import yaml
import cv2
import numpy as np
from ultralytics import YOLO

def check_data_yaml(data_path="chip_defects.yaml"):
    """Проверка конфигурационного файла данных"""
    print("=== Проверка конфигурационного файла данных ===")
    
    if not os.path.exists(data_path):
        print(f"❌ Файл {data_path} не найден")
        return False
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        print(f"✅ Файл {data_path} успешно загружен")
        print(f"Пути к данным:")
        print(f"  - train: {data.get('train', 'не указан')}")
        print(f"  - val: {data.get('val', 'не указан')}")
        print(f"  - test: {data.get('test', 'не указан')}")
        
        print(f"Количество классов: {data.get('nc', 'не указано')}")
        print(f"Имена классов: {data.get('names', 'не указаны')}")
        
        # Проверка существования путей
        for key in ['train', 'val', 'test']:
            if key in data:
                path = data[key]
                if os.path.exists(path):
                    print(f"✅ Путь {key}: {path} существует")
                else:
                    print(f"❌ Путь {key}: {path} не существует")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при чтении {data_path}: {e}")
        return False

def check_model_file(model_path):
    """Проверка файла модели"""
    print("\n=== Проверка файла модели ===")
    
    if not os.path.exists(model_path):
        print(f"❌ Файл модели {model_path} не найден")
        return False
    
    try:
        # Попробуем загрузить модель
        model = YOLO(model_path)
        print(f"✅ Модель {model_path} успешно загружена")
        
        # Проверим информацию о модели
        print(f"Архитектура модели: {model.__class__.__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели {model_path}: {e}")
        return False

def check_sample_images(data_path="chip_defects.yaml"):
    """Проверка образцов изображений и их разметки"""
    print("\n=== Проверка образцов изображений ===")
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        val_path = data.get('val')
        if not val_path or not os.path.exists(val_path):
            print("❌ Путь к валидационным данным не найден")
            return
        
        # Получим список изображений
        if os.path.isfile(val_path):
            # Если указан конкретный файл
            image_files = [val_path]
        else:
            # Если указана директория
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                import glob
                image_files.extend(glob.glob(os.path.join(val_path, ext)))
        
        if not image_files:
            print("❌ Изображения не найдены")
            return
        
        # Проверим первые 5 изображений
        print(f"Найдено {len(image_files)} изображений. Проверяем первые 5:")
        
        for i, img_path in enumerate(image_files[:5]):
            print(f"\nИзображение {i+1}: {os.path.basename(img_path)}")
            
            # Проверим изображение
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    print(f"  ✅ Изображение загружено: {img.shape[1]}x{img.shape[0]}")
                else:
                    print(f"  ❌ Не удалось загрузить изображение")
                    continue
            else:
                print(f"  ❌ Файл изображения не найден")
                continue
            
            # Проверим разметку (.txt файл)
            txt_path = os.path.splitext(img_path)[0] + '.txt'
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, 'r') as f:
                        lines = f.readlines()
                    
                    print(f"  ✅ Найден файл разметки с {len(lines)} объектами")
                    
                    # Проверим формат разметки
                    for j, line in enumerate(lines):
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            print(f"    Объект {j+1}: класс {class_id}, bbox ({x_center:.3f}, {y_center:.3f}, {width:.3f}, {height:.3f})")
                        else:
                            print(f"    ❌ Неправильный формат строки {j+1}: {line}")
                except Exception as e:
                    print(f"  ❌ Ошибка при чтении разметки: {e}")
            else:
                print(f"  ❌ Файл разметки не найден: {txt_path}")
        
    except Exception as e:
        print(f"❌ Ошибка при проверке изображений: {e}")

def check_training_logs():
    """Проверка логов тренировки"""
    print("\n=== Проверка логов тренировки ===")
    
    # Пути к возможным логам тренировки
    log_paths = [
        "runs/detect/train_optimized/results.csv",
        "runs/detect/train5/results.csv",
        "runs/detect/train/results.csv"
    ]
    
    found_logs = False
    for log_path in log_paths:
        if os.path.exists(log_path):
            print(f"✅ Найдены логи тренировки: {log_path}")
            found_logs = True
            
            # Покажем последние строки лога
            try:
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    print("Последние 10 строк лога:")
                    for line in lines[-10:]:
                        print(f"  {line.strip()}")
            except Exception as e:
                print(f"❌ Ошибка при чтении лога: {e}")
    
    if not found_logs:
        print("❌ Логи тренировки не найдены")

def diagnose_model(model_path=None, data_path="chip_defects.yaml"):
    """
    Полная диагностика модели и данных
    
    Args:
        model_path (str): Путь к файлу модели (.pt)
        data_path (str): Путь к конфигурационному файлу данных (.yaml)
    """
    print("Диагностика модели и данных для YOLO")
    print("=" * 50)
    
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
            print("❌ Модель не найдена. Пожалуйста, сначала натренируйте модель.")
            print("Ожидаемые пути к модели:")
            for path in model_paths:
                print(f"  - {path}")
            return
    
    # Выполнение проверок
    data_ok = check_data_yaml(data_path)
    model_ok = check_model_file(model_path)
    
    if data_ok:
        check_sample_images(data_path)
    
    check_training_logs()
    
    print("\n" + "=" * 50)
    if data_ok and model_ok:
        print("✅ Базовая диагностика пройдена. Для более детальной проверки используйте валидацию.")
    else:
        print("❌ Обнаружены проблемы. Проверьте сообщения выше.")

def main():
    """Основная функция"""
    # Активация виртуального окружения при необходимости
    venv_path = os.path.join(os.path.dirname(__file__), '..', 'venv')
    if os.path.exists(venv_path):
        site_packages = os.path.join(venv_path, 'Lib', 'site-packages')
        if site_packages not in sys.path:
            sys.path.insert(0, site_packages)
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Диагностика проблем с моделью и данными")
    parser.add_argument("--model", type=str, help="Путь к файлу модели (.pt)")
    parser.add_argument("--data", type=str, default="chip_defects.yaml", help="Путь к конфигурационному файлу данных (.yaml)")
    
    args = parser.parse_args()
    
    diagnose_model(model_path=args.model, data_path=args.data)

if __name__ == "__main__":
    main()