#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для проверки совместимости версий CUDA, cuDNN и PyTorch
"""

import sys
import subprocess
import platform

def get_cuda_version():
    """Получение версии CUDA"""
    try:
        # Попробуем получить версию CUDA через nvcc
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, check=True)
        lines = result.stdout.split('\n')
        for line in lines:
            if 'release' in line:
                # Извлекаем версию из строки вида "release 11.8, V11.8.89"
                parts = line.split(',')
                if len(parts) > 0:
                    version_part = parts[0]
                    version = version_part.split()[-1]
                    return version
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return None

def get_cudnn_version():
    """Получение версии cuDNN через PyTorch"""
    try:
        import torch
        if torch.cuda.is_available():
            cudnn_version = torch.backends.cudnn.version()
            return cudnn_version
    except (ImportError, AttributeError):
        pass
    return None

def get_pytorch_version():
    """Получение версии PyTorch"""
    try:
        import torch
        return torch.__version__
    except ImportError:
        return None

def check_compatibility():
    """Проверка совместимости версий"""
    print("=== Проверка совместимости версий ===")
    
    cuda_version = get_cuda_version()
    cudnn_version = get_cudnn_version()
    pytorch_version = get_pytorch_version()
    
    print(f"Установленная версия PyTorch: {pytorch_version or 'Не установлена'}")
    print(f"Установленная версия CUDA: {cuda_version or 'Не установлена'}")
    print(f"Установленная версия cuDNN: {cudnn_version or 'Не установлена'}")
    print()
    
    # Проверка совместимости
    compatibility_issues = []
    
    if pytorch_version is None:
        compatibility_issues.append("PyTorch не установлен")
    else:
        print("=== Совместимость PyTorch и CUDA ===")
        # PyTorch 2.9.0 совместим с CUDA 11.8 и 12.1
        if cuda_version:
            if cuda_version.startswith('11.8'):
                print("✓ PyTorch 2.9.0 совместим с CUDA 11.8")
            elif cuda_version.startswith('12.1'):
                print("✓ PyTorch 2.9.0 совместим с CUDA 12.1")
            else:
                print(f"⚠ PyTorch 2.9.0 может быть несовместим с CUDA {cuda_version}")
                print("  Рекомендуется использовать CUDA 11.8 или 12.1")
                compatibility_issues.append(f"Несовместимая версия CUDA: {cuda_version}")
        else:
            print("⚠ CUDA не установлена")
            compatibility_issues.append("CUDA не установлена")
    
    if cudnn_version and cuda_version:
        print("\n=== Совместимость cuDNN и CUDA ===")
        # Проверка совместимости cuDNN и CUDA
        if cuda_version.startswith('11.8') and str(cudnn_version).startswith('87'):
            print("✓ cuDNN совместим с CUDA 11.8")
        elif cuda_version.startswith('12.1') and str(cudnn_version).startswith('89'):
            print("✓ cuDNN совместим с CUDA 12.1")
        else:
            print(f"⚠ Возможна несовместимость cuDNN {cudnn_version} с CUDA {cuda_version}")
            print("  Рекомендуется проверить таблицу совместимости NVIDIA")
            compatibility_issues.append(f"Несовместимость cuDNN {cudnn_version} и CUDA {cuda_version}")
    
    print("\n=== Рекомендации ===")
    if compatibility_issues:
        print("Найдены проблемы совместимости:")
        for issue in compatibility_issues:
            print(f"  - {issue}")
        print("\nРекомендуется:")
        if "PyTorch не установлен" in compatibility_issues:
            print("  1. Установить PyTorch с поддержкой CUDA:")
            print("     pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu118")
        if "CUDA не установлена" in compatibility_issues:
            print("  2. Установить CUDA Toolkit 11.8 или 12.1")
        if "Несовместимая версия CUDA" in compatibility_issues:
            print("  3. Установить совместимую версию CUDA Toolkit")
    else:
        print("✓ Все компоненты совместимы")
        print("\nДля проверки работы GPU с YOLO выполните:")
        print("  python -c \"import torch; print('CUDA доступна:', torch.cuda.is_available())\"")
        print("  yolo detect train ... --device 0  # для использования GPU")

def check_environment_variables():
    """Проверка переменных окружения"""
    print("\n=== Проверка переменных окружения ===")
    import os
    
    cuda_path = os.environ.get('CUDA_PATH')
    path = os.environ.get('PATH')
    
    if cuda_path:
        print(f"CUDA_PATH: {cuda_path}")
    else:
        print("CUDA_PATH не установлена")
    
    if path:
        cuda_in_path = any('cuda' in p.lower() for p in path.split(os.pathsep))
        if cuda_in_path:
            print("Пути к CUDA найдены в PATH")
        else:
            print("Пути к CUDA НЕ найдены в PATH")
            print("Рекомендуется добавить пути к CUDA в переменную среды PATH")

def main():
    """Основная функция"""
    print("Проверка совместимости CUDA, cuDNN и PyTorch")
    print("=" * 50)
    
    check_compatibility()
    check_environment_variables()
    
    print("\nДля получения подробной информации выполните:")
    print("python gpu_diagnostic.py")

if __name__ == "__main__":
    main()