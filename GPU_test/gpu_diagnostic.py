#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для диагностики конфигурации GPU и проверки доступности необходимых библиотек
"""

import sys
import subprocess
import platform

def check_python_version():
    """Проверка версии Python"""
    print("=== Информация о Python ===")
    print(f"Версия Python: {sys.version}")
    print(f"Архитектура: {platform.architecture()}")
    print(f"Платформа: {platform.platform()}")
    print()

def check_cuda():
    """Проверка наличия CUDA"""
    print("=== Информация о CUDA ===")
    try:
        # Проверка версии CUDA через nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,cuda_version', '--format=csv'],
                                capture_output=True, text=True, check=True)
        print("Информация о GPU от nvidia-smi:")
        print(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("nvidia-smi не найден. Убедитесь, что драйверы NVIDIA установлены.")
    
    try:
        # Проверка версии CUDA через nvcc
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, check=True)
        print("Версия CUDA компилятора:")
        print(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("nvcc не найден. CUDA Toolkit может быть не установлен.")
    print()

def check_pytorch():
    """Проверка PyTorch и доступности GPU"""
    print("=== Информация о PyTorch ===")
    try:
        import torch
        print(f"Версия PyTorch: {torch.__version__}")
        print(f"Доступен CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Количество GPU: {torch.cuda.device_count()}")
            print(f"Текущий GPU: {torch.cuda.current_device()}")
            print(f"Имя GPU: {torch.cuda.get_device_name()}")
            print(f"Версия CUDA: {torch.version.cuda}")
            print(f"Версия cuDNN: {torch.backends.cudnn.version()}")
        else:
            print("CUDA недоступна для PyTorch")
    except ImportError:
        print("PyTorch не установлен")
    print()

def check_tensorflow():
    """Проверка TensorFlow и доступности GPU"""
    print("=== Информация о TensorFlow ===")
    try:
        import tensorflow as tf
        print(f"Версия TensorFlow: {tf.__version__}")
        print(f"Доступен GPU: {len(tf.config.experimental.list_physical_devices('GPU')) > 0}")
        if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
            print(f"Количество доступных GPU: {len(tf.config.experimental.list_physical_devices('GPU'))}")
            for i, gpu in enumerate(tf.config.experimental.list_physical_devices('GPU')):
                print(f"GPU {i}: {gpu}")
        else:
            print("GPU недоступен для TensorFlow")
    except ImportError:
        print("TensorFlow не установлен")
    print()

def check_pip_packages():
    """Проверка установленных пакетов через pip"""
    print("=== Установленные пакеты (выборочно) ===")
    packages_to_check = ['torch', 'torchvision', 'tensorflow', 'opencv-python', 'numpy', 'cuda-python']
    
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], capture_output=True, text=True, check=True)
        installed_packages = result.stdout
        
        for package in packages_to_check:
            if package in installed_packages:
                # Найти строку с пакетом
                for line in installed_packages.split('\n'):
                    if line.startswith(package):
                        print(line)
                        break
            else:
                print(f"{package}: не установлен")
    except subprocess.CalledProcessError:
        print("Не удалось получить список установленных пакетов")
    print()

def main():
    """Основная функция"""
    print("Диагностика конфигурации GPU для YOLO")
    print("=" * 50)
    
    check_python_version()
    check_cuda()
    check_pytorch()
    check_tensorflow()
    check_pip_packages()
    
    print("Диагностика завершена.")

if __name__ == "__main__":
    main()