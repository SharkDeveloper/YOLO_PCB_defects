#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для установки CUDA Toolkit, совместимого с PyTorch 2.9.0
"""

import sys
import subprocess
import platform
import webbrowser

def print_cuda_info():
    """Вывод информации о совместимых версиях CUDA для PyTorch 2.9.0"""
    print("=== Совместимость CUDA и PyTorch 2.9.0 ===")
    print("PyTorch 2.9.0 совместим со следующими версиями CUDA:")
    print("  - CUDA 11.8")
    print("  - CUDA 12.1")
    print()
    print("Рекомендуется установить CUDA 11.8 для максимальной совместимости")
    print()

def check_os():
    """Проверка операционной системы"""
    system = platform.system()
    arch = platform.machine()
    
    print("=== Информация о системе ===")
    print(f"Операционная система: {system}")
    print(f"Архитектура: {arch}")
    print()
    
    return system, arch

def install_cuda_windows():
    """Инструкции по установке CUDA на Windows"""
    print("=== Установка CUDA Toolkit на Windows ===")
    print("1. Перейдите на официальный сайт NVIDIA CUDA Toolkit:")
    print("   https://developer.nvidia.com/cuda-11-8-0-download-archive")
    print()
    print("2. Выберите параметры для загрузки:")
    print("   - Operating System: Windows")
    print("   - Architecture: x86_64")
    print("   - Version: 10 или 11")
    print("   - Installer Type: exe (local)")
    print()
    print("3. Загрузите и запустите установщик")
    print("4. Следуйте инструкциям установщика")
    print("5. Перезагрузите компьютер после установки")
    print()
    
    # Открыть ссылку в браузере
    try:
        webbrowser.open("https://developer.nvidia.com/cuda-11-8-0-download-archive")
        print("Открыта страница загрузки CUDA 11.8 в браузере")
    except:
        print("Не удалось открыть браузер. Пожалуйста, перейдите по ссылке вручную")

def install_cuda_linux():
    """Инструкции по установке CUDA на Linux"""
    print("=== Установка CUDA Toolkit на Linux ===")
    print("Для Ubuntu/Debian:")
    print("1. Загрузите репозиторий CUDA:")
    print("   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb")
    print("   sudo dpkg -i cuda-keyring_1.0-1_all.deb")
    print("   sudo apt-get update")
    print()
    print("2. Установите CUDA Toolkit:")
    print("   sudo apt-get install cuda-toolkit-11-8")
    print()
    print("3. Добавьте CUDA в PATH (добавьте в ~/.bashrc):")
    print('   export PATH=/usr/local/cuda-11.8/bin:$PATH')
    print('   export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH')
    print()
    print("4. Перезагрузите компьютер или выполните:")
    print("   source ~/.bashrc")

def install_cuda_mac():
    """Инструкции по установке CUDA на macOS"""
    print("=== Установка CUDA Toolkit на macOS ===")
    print("ВНИМАНИЕ: NVIDIA прекратила поддержку CUDA на macOS после CUDA 10.2")
    print("Для macOS рекомендуется использовать CPU или облачные решения")
    print()
    print("Альтернативы для macOS:")
    print("1. Использование Google Colab с GPU")
    print("2. Использование облачных сервисов (AWS, GCP, Azure)")
    print("3. Использование Apple Silicon (M1/M2) с Metal Performance Shaders")

def main():
    """Основная функция"""
    print("Установка CUDA Toolkit для PyTorch 2.9.0")
    print("=" * 50)
    
    print_cuda_info()
    system, arch = check_os()
    
    if system == "Windows":
        install_cuda_windows()
    elif system == "Linux":
        install_cuda_linux()
    elif system == "Darwin":  # macOS
        install_cuda_mac()
    else:
        print(f"Неизвестная операционная система: {system}")
        print("Пожалуйста, посетите https://developer.nvidia.com/cuda-downloads для получения инструкций")
    
    print()
    print("После установки CUDA Toolkit перезапустите терминал и выполните:")
    print("python gpu_diagnostic.py")
    print("для проверки установки")

if __name__ == "__main__":
    main()