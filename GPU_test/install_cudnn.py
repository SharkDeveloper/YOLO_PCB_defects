#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для установки cuDNN, совместимого с CUDA Toolkit
"""

import sys
import subprocess
import platform
import webbrowser
import os

def print_cudnn_info():
    """Вывод информации о cuDNN и его совместимости"""
    print("=== Информация о cuDNN ===")
    print("cuDNN (CUDA Deep Neural Network library) - библиотека примитивов для глубокого обучения")
    print("cuDNN должен быть совместим с установленной версией CUDA Toolkit")
    print()
    print("Для PyTorch 2.9.0 рекомендуются следующие версии cuDNN:")
    print("  - cuDNN 8.7.x (для CUDA 11.8)")
    print("  - cuDNN 8.9.x (для CUDA 12.1)")
    print()

def check_cuda_version():
    """Проверка установленной версии CUDA"""
    print("=== Проверка версии CUDA ===")
    try:
        # Попробуем получить версию CUDA через nvcc
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, check=True)
        lines = result.stdout.split('\n')
        for line in lines:
            if 'release' in line:
                print(f"Обнаруженная версия CUDA: {line.strip()}")
                return line.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("nvcc не найден. CUDA может быть не установлена или не добавлена в PATH")
        print("Пожалуйста, сначала установите CUDA Toolkit")
        return None
    
    return None

def install_cudnn_windows(cuda_version):
    """Инструкции по установке cuDNN на Windows"""
    print("=== Установка cuDNN на Windows ===")
    
    if '11.8' in cuda_version:
        cudnn_version = "8.7.x"
    elif '12.1' in cuda_version:
        cudnn_version = "8.9.x"
    else:
        cudnn_version = "8.x"
    
    print(f"Рекомендуемая версия cuDNN: {cudnn_version}")
    print()
    print("1. Перейдите на официальный сайт NVIDIA cuDNN:")
    print("   https://developer.nvidia.com/rdp/cudnn-archive")
    print()
    print("2. Войдите в учетную запись NVIDIA или зарегистрируйтесь")
    print("3. Выберите подходящую версию cuDNN для вашей CUDA:")
    if '11.8' in cuda_version:
        print("   - cuDNN v8.7.0 (January 26th, 2023), for CUDA 11.8")
    elif '12.1' in cuda_version:
        print("   - cuDNN v8.9.0 (April 6th, 2023), for CUDA 12.1")
    print()
    print("4. Загрузите архив cuDNN для вашей операционной системы")
    print("5. Распакуйте архив в папку, например: C:\\tools\\cudnn")
    print("6. Добавьте путь к библиотекам cuDNN в переменную среды PATH:")
    print("   - Откройте Панель управления -> Система -> Дополнительные параметры системы")
    print("   - Нажмите 'Переменные среды'")
    print("   - В разделе 'Системные переменные' найдите и выберите 'Path', нажмите 'Изменить'")
    print("   - Добавьте путь к папке bin из распакованного cuDNN:")
    print("     Например: C:\\tools\\cudnn\\bin")
    print("7. Перезапустите терминал/командную строку")
    print()
    
    # Открыть ссылку в браузере
    try:
        webbrowser.open("https://developer.nvidia.com/rdp/cudnn-archive")
        print("Открыта страница загрузки cuDNN в браузере")
    except:
        print("Не удалось открыть браузер. Пожалуйста, перейдите по ссылке вручную")

def install_cudnn_linux(cuda_version):
    """Инструкции по установке cuDNN на Linux"""
    print("=== Установка cuDNN на Linux ===")
    
    if '11.8' in cuda_version:
        cudnn_version = "8.7.x"
    elif '12.1' in cuda_version:
        cudnn_version = "8.9.x"
    else:
        cudnn_version = "8.x"
    
    print(f"Рекомендуемая версия cuDNN: {cudnn_version}")
    print()
    print("Для Ubuntu/Debian:")
    print("1. Перейдите на официальный сайт NVIDIA cuDNN:")
    print("   https://developer.nvidia.com/rdp/cudnn-archive")
    print()
    print("2. Войдите в учетную запись NVIDIA или зарегистрируйтесь")
    print("3. Выберите подходящую версию cuDNN для вашей CUDA:")
    if '11.8' in cuda_version:
        print("   - cuDNN v8.7.0 (January 26th, 2023), for CUDA 11.8")
    elif '12.1' in cuda_version:
        print("   - cuDNN v8.9.0 (April 6th, 2023), for CUDA 12.1")
    print()
    print("4. Загрузите следующие пакеты .deb:")
    print("   - cuDNN Runtime Library for Ubuntu20.04 x86_64 (Deb)")
    print("   - cuDNN Developer Library for Ubuntu20.04 x86_64 (Deb)")
    print()
    print("5. Установите пакеты:")
    print("   sudo dpkg -i libcudnn8_*.deb")
    print("   sudo dpkg -i libcudnn8-dev_*.deb")
    print()
    print("Альтернативно, можно установить через tar-файл:")
    print("1. Загрузите tar-файл cuDNN для Linux")
    print("2. Распакуйте архив:")
    print("   tar -xzvf cudnn-linux-x86_64-8.x.x.x_cudaX.X-archive.tar.xz")
    print("3. Скопируйте файлы в директорию CUDA:")
    print("   sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include")
    print("   sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64")
    print("   sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*")

def install_cudnn_mac():
    """Инструкции по установке cuDNN на macOS"""
    print("=== Установка cuDNN на macOS ===")
    print("ВНИМАНИЕ: NVIDIA прекратила поддержку CUDA на macOS после CUDA 10.2")
    print("Следовательно, cuDNN также не поддерживается на macOS")
    print()
    print("Альтернативы для macOS:")
    print("1. Использование Google Colab с GPU")
    print("2. Использование облачных сервисов (AWS, GCP, Azure)")
    print("3. Использование Apple Silicon (M1/M2) с Metal Performance Shaders")

def verify_cudnn_installation():
    """Проверка установки cuDNN"""
    print("=== Проверка установки cuDNN ===")
    print("После установки cuDNN выполните следующие команды для проверки:")
    print()
    print("Для PyTorch:")
    print("python -c \"import torch; print(f'cuDNN доступен: {torch.backends.cudnn.enabled}'); print(f'Версия cuDNN: {torch.backends.cudnn.version()}')\"")
    print()
    print("Для TensorFlow:")
    print("python -c \"import tensorflow as tf; print('GPU доступен:', tf.config.list_physical_devices('GPU'))\"")

def main():
    """Основная функция"""
    print("Установка cuDNN для CUDA Toolkit")
    print("=" * 50)
    
    print_cudnn_info()
    
    # Проверим версию CUDA
    cuda_version = check_cuda_version()
    
    if cuda_version is None:
        print("Продолжить установку cuDNN без информации о версии CUDA? (y/n): ", end="")
        response = input().strip().lower()
        if response != 'y':
            print("Установка прервана. Пожалуйста, сначала установите CUDA Toolkit.")
            return
    
    system = platform.system()
    
    if system == "Windows":
        install_cudnn_windows(cuda_version or "")
    elif system == "Linux":
        install_cudnn_linux(cuda_version or "")
    elif system == "Darwin":  # macOS
        install_cudnn_mac()
    else:
        print(f"Неизвестная операционная система: {system}")
        print("Пожалуйста, посетите https://developer.nvidia.com/cudnn для получения инструкций")
    
    print()
    verify_cudnn_installation()
    print()
    print("После установки cuDNN перезапустите терминал и выполните:")
    print("python gpu_diagnostic.py")
    print("для проверки установки")

if __name__ == "__main__":
    main()