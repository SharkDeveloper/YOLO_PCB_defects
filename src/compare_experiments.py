#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для сравнения метрик разных экспериментов
"""

import os
import pandas as pd
import argparse

def compare_experiments(metrics_file="metrics/summary.csv"):
    """
    Сравнение метрик разных экспериментов
    
    Args:
        metrics_file (str): Путь к файлу с метриками
    """
    if not os.path.exists(metrics_file):
        print(f"Файл с метриками {metrics_file} не найден.")
        print("Пожалуйста, сначала запустите обучение, валидацию или инференс.")
        return
    
    try:
        # Чтение данных
        df = pd.read_csv(metrics_file)
        
        if df.empty:
            print("Файл с метриками пуст.")
            return
        
        # Вывод всех экспериментов
        print("Сравнение экспериментов:")
        print("=" * 100)
        
        # Форматированный вывод
        print(f"{'Эксперимент':<25} {'Тип':<10} {'Время обучения':<15} {'mAP50 (train)':<15} {'Время валидации':<15} {'mAP50 (val)':<15} {'Время инференса':<15} {'mAP50 (inf)':<15}")
        print("-" * 100)
        
        for _, row in df.iterrows():
            # Обработка NaN значений
            def format_value(val):
                if pd.isna(val) or val == '':
                    return 'N/A'
                # Если значение число, округляем до 4 знаков
                try:
                    return f"{float(val):.4f}"
                except:
                    return str(val)
            
            print(f"{row['experiment_name']:<25} "
                  f"{format_value(row['model_type']):<10} "
                  f"{format_value(row['training_time']):<15} "
                  f"{format_value(row['training_mAP50']):<15} "
                  f"{format_value(row['validation_time']):<15} "
                  f"{format_value(row['validation_mAP50']):<15} "
                  f"{format_value(row['inference_time']):<15} "
                  f"{format_value(row['inference_mAP50']):<15}")
        
        print("-" * 100)
        print(f"Всего экспериментов: {len(df)}")
        
    except Exception as e:
        print(f"Ошибка при чтении файла метрик: {e}")

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description="Сравнение метрик разных экспериментов")
    parser.add_argument("--metrics-file", type=str, default="metrics/summary.csv", 
                        help="Путь к файлу с метриками (по умолчанию: metrics/summary.csv)")
    
    args = parser.parse_args()
    
    compare_experiments(args.metrics_file)

if __name__ == "__main__":
    main()