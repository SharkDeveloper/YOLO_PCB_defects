import os
import json
import csv
import time
from datetime import datetime
import pandas as pd
from ultralytics import YOLO


# Путь к директории для сохранения метрик
METRICS_DIR = "metrics"

def init_metrics_logging():
    """Инициализация директории для логирования метрик"""
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    # Создание файла для общей сводки метрик, если он не существует
    summary_file = os.path.join(METRICS_DIR, "summary.csv")
    if not os.path.exists(summary_file):
        with open(summary_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'experiment_name', 'model_type', 
                'training_time', 'training_speed', 'training_precision', 'training_recall', 'training_mAP50', 'training_mAP50-95',
                'validation_time', 'validation_speed', 'validation_precision', 'validation_recall', 'validation_mAP50', 'validation_mAP50-95',
                'inference_time', 'inference_speed', 'inference_precision', 'inference_recall', 'inference_mAP50', 'inference_mAP50-95'
            ])
    
    return summary_file

def log_training_metrics(experiment_name, model_type, start_time, end_time, metrics_data=None):
    """
    Логирование метрик обучения
    
    Args:
        experiment_name (str): Название эксперимента
        model_type (str): Тип модели
        start_time (float): Время начала обучения
        end_time (float): Время окончания обучения
        metrics_data (dict): Дополнительные метрики обучения
    """
    training_time = end_time - start_time
    
    # Инициализация логирования
    summary_file = init_metrics_logging()
    
    # Чтение существующих данных
    summary_data = []
    if os.path.exists(summary_file):
        try:
            df = pd.read_csv(summary_file)
            summary_data = df.to_dict('records')
        except:
            pass
    
    # Поиск существующей записи для этого эксперимента
    experiment_found = False
    for record in summary_data:
        if record['experiment_name'] == experiment_name:
            # Обновление данных обучения
            record['training_time'] = training_time
            record['model_type'] = model_type
            if metrics_data:
                record['training_speed'] = metrics_data.get('speed', '')
                record['training_precision'] = metrics_data.get('precision', '')
                record['training_recall'] = metrics_data.get('recall', '')
                record['training_mAP50'] = metrics_data.get('mAP50', '')
                record['training_mAP50-95'] = metrics_data.get('mAP50-95', '')
            experiment_found = True
            break
    
    # Если эксперимент не найден, создаем новую запись
    if not experiment_found:
        new_record = {
            'timestamp': datetime.now().isoformat(),
            'experiment_name': experiment_name,
            'model_type': model_type,
            'training_time': training_time,
            'training_speed': metrics_data.get('speed', '') if metrics_data else '',
            'training_precision': metrics_data.get('precision', '') if metrics_data else '',
            'training_recall': metrics_data.get('recall', '') if metrics_data else '',
            'training_mAP50': metrics_data.get('mAP50', '') if metrics_data else '',
            'training_mAP50-95': metrics_data.get('mAP50-95', '') if metrics_data else '',
            # Остальные поля оставляем пустыми
            'validation_time': '',
            'validation_speed': '',
            'validation_precision': '',
            'validation_recall': '',
            'validation_mAP50': '',
            'validation_mAP50-95': '',
            'inference_time': '',
            'inference_speed': '',
            'inference_precision': '',
            'inference_recall': '',
            'inference_mAP50': '',
            'inference_mAP50-95': ''
        }
        summary_data.append(new_record)
    
    # Сохранение обновленных данных
    df = pd.DataFrame(summary_data)
    df.to_csv(summary_file, index=False, encoding='utf-8')
    
    print(f"Метрики обучения сохранены в {summary_file}")

def log_validation_metrics(experiment_name, start_time, end_time, metrics_data=None):
    """
    Логирование метрик валидации
    
    Args:
        experiment_name (str): Название эксперимента
        start_time (float): Время начала валидации
        end_time (float): Время окончания валидации
        metrics_data (dict): Метрики валидации
    """
    validation_time = end_time - start_time
    
    # Инициализация логирования
    summary_file = init_metrics_logging()
    
    # Чтение существующих данных
    summary_data = []
    if os.path.exists(summary_file):
        try:
            df = pd.read_csv(summary_file)
            summary_data = df.to_dict('records')
        except:
            pass
    
    # Поиск существующей записи для этого эксперимента
    experiment_found = False
    for record in summary_data:
        if record['experiment_name'] == experiment_name:
            # Обновление данных валидации
            record['validation_time'] = validation_time
            if metrics_data:
                record['validation_speed'] = metrics_data.get('speed', '')
                record['validation_precision'] = metrics_data.get('precision', '')
                record['validation_recall'] = metrics_data.get('recall', '')
                record['validation_mAP50'] = metrics_data.get('mAP50', '')
                record['validation_mAP50-95'] = metrics_data.get('mAP50-95', '')
            experiment_found = True
            break
    
    # Если эксперимент не найден, создаем новую запись
    if not experiment_found:
        new_record = {
            'timestamp': datetime.now().isoformat(),
            'experiment_name': experiment_name,
            'model_type': '',
            'training_time': '',
            'training_speed': '',
            'training_precision': '',
            'training_recall': '',
            'training_mAP50': '',
            'training_mAP50-95': '',
            'validation_time': validation_time,
            'validation_speed': metrics_data.get('speed', '') if metrics_data else '',
            'validation_precision': metrics_data.get('precision', '') if metrics_data else '',
            'validation_recall': metrics_data.get('recall', '') if metrics_data else '',
            'validation_mAP50': metrics_data.get('mAP50', '') if metrics_data else '',
            'validation_mAP50-95': metrics_data.get('mAP50-95', '') if metrics_data else '',
            # Остальные поля оставляем пустыми
            'inference_time': '',
            'inference_speed': '',
            'inference_precision': '',
            'inference_recall': '',
            'inference_mAP50': '',
            'inference_mAP50-95': ''
        }
        summary_data.append(new_record)
    
    # Сохранение обновленных данных
    df = pd.DataFrame(summary_data)
    df.to_csv(summary_file, index=False, encoding='utf-8')
    
    print(f"Метрики валидации сохранены в {summary_file}")

def log_inference_metrics(experiment_name, start_time, end_time, metrics_data=None):
    """
    Логирование метрик инференса
    
    Args:
        experiment_name (str): Название эксперимента
        start_time (float): Время начала инференса
        end_time (float): Время окончания инференса
        metrics_data (dict): Метрики инференса
    """
    inference_time = end_time - start_time
    
    # Инициализация логирования
    summary_file = init_metrics_logging()
    
    # Чтение существующих данных
    summary_data = []
    if os.path.exists(summary_file):
        try:
            df = pd.read_csv(summary_file)
            summary_data = df.to_dict('records')
        except:
            pass
    
    # Поиск существующей записи для этого эксперимента
    experiment_found = False
    for record in summary_data:
        if record['experiment_name'] == experiment_name:
            # Обновление данных инференса
            record['inference_time'] = inference_time
            if metrics_data:
                record['inference_speed'] = metrics_data.get('speed', '')
                record['inference_precision'] = metrics_data.get('precision', '')
                record['inference_recall'] = metrics_data.get('recall', '')
                record['inference_mAP50'] = metrics_data.get('mAP50', '')
                record['inference_mAP50-95'] = metrics_data.get('mAP50-95', '')
            experiment_found = True
            break
    
    # Если эксперимент не найден, создаем новую запись
    if not experiment_found:
        new_record = {
            'timestamp': datetime.now().isoformat(),
            'experiment_name': experiment_name,
            'model_type': '',
            'training_time': '',
            'training_speed': '',
            'training_precision': '',
            'training_recall': '',
            'training_mAP50': '',
            'training_mAP50-95': '',
            'validation_time': '',
            'validation_speed': '',
            'validation_precision': '',
            'validation_recall': '',
            'validation_mAP50': '',
            'validation_mAP50-95': '',
            'inference_time': inference_time,
            'inference_speed': metrics_data.get('speed', '') if metrics_data else '',
            'inference_precision': metrics_data.get('precision', '') if metrics_data else '',
            'inference_recall': metrics_data.get('recall', '') if metrics_data else '',
            'inference_mAP50': metrics_data.get('mAP50', '') if metrics_data else '',
            'inference_mAP50-95': metrics_data.get('mAP50-95', '') if metrics_data else ''
        }
        summary_data.append(new_record)
    
    # Сохранение обновленных данных
    df = pd.DataFrame(summary_data)
    df.to_csv(summary_file, index=False, encoding='utf-8')
    
    print(f"Метрики инференса сохранены в {summary_file}")

def parse_yolo_output(output_lines):
    """
    Парсинг вывода YOLO для извлечения метрик
    
    Args:
        output_lines (list): Строки вывода YOLO
        
    Returns:
        dict: Словарь с извлеченными метриками
    """
    metrics = {}
    
    # Поиск метрик в выводе
    for line in output_lines:
        line = line.strip()
        
        # Поиск скорости (Speed)
        if 'Speed:' in line:
            # Пример: "Speed: 0.1ms preprocess, 5.2ms inference, 0.3ms postprocess per image"
            metrics['speed'] = line
        
        # Поиск метрик точности (в строках валидации)
        if 'Precision:' in line:
            # Пример: "Precision: 0.85"
            parts = line.split()
            for i, part in enumerate(parts):
                if part == 'Precision:':
                    if i + 1 < len(parts):
                        try:
                            metrics['precision'] = float(parts[i + 1])
                        except:
                            metrics['precision'] = parts[i + 1]
        
        if 'Recall:' in line:
            # Пример: "Recall: 0.78"
            parts = line.split()
            for i, part in enumerate(parts):
                if part == 'Recall:':
                    if i + 1 < len(parts):
                        try:
                            metrics['recall'] = float(parts[i + 1])
                        except:
                            metrics['recall'] = parts[i + 1]
        
        if 'mAP50:' in line:
            # Пример: "mAP50: 0.82"
            parts = line.split()
            for i, part in enumerate(parts):
                if part == 'mAP50:':
                    if i + 1 < len(parts):
                        try:
                            metrics['mAP50'] = float(parts[i + 1])
                        except:
                            metrics['mAP50'] = parts[i + 1]
        
        if 'mAP50-95:' in line:
            # Пример: "mAP50-95: 0.65"
            parts = line.split()
            for i, part in enumerate(parts):
                if part == 'mAP50-95:':
                    if i + 1 < len(parts):
                        try:
                            metrics['mAP50-95'] = float(parts[i + 1])
                        except:
                            metrics['mAP50-95'] = parts[i + 1]
    
    return metrics

def get_experiment_name(base_name=None):
    """
    Генерация уникального имени эксперимента
    
    Args:
        base_name (str): Базовое имя эксперимента
        
    Returns:
        str: Уникальное имя эксперимента
    """
    if base_name is None:
        base_name = "experiment"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}"


def convert_to_onnx(model_path, output_path):

    # Load a YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Export the model
    model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=640)
