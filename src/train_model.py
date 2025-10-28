import sys
import os
import time
import torch
from ultralytics import YOLO
from utils import parse_yolo_output, log_training_metrics, get_experiment_name

def check_cuda():
    """Check if CUDA is available and print GPU information"""
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        return True
    else:
        print("CUDA is not available. Training will use CPU (slower).")
        print("To speed up training, please install CUDA-compatible PyTorch version.")
        return False

def train_model():
    """Train YOLOv8 model for PCB defect detection with optimized parameters using direct API"""
    # Check CUDA availability
    cuda_available = check_cuda()

    # Generate experiment name
    experiment_name = get_experiment_name("train")

    # Set training parameters as dict
    train_params = {
        'data': 'chip_defects.yaml',
        'epochs': 10,
        'imgsz': 640,
        'batch': 16,
        'device': 0 if cuda_available else 'cpu',
        'amp': False,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'patience': 1,  # Early stopping patience (uncomment if needed)
        'workers': 6,
        'project': 'runs/detect',
        'name': 'train_optimized',
        'exist_ok': True,
        # можно добавлять другие параметры
    }

    print("Training parameters:")
    for k, v in train_params.items():
        print(f"{k}: {v}")

    # Start time measurement
    start_time = time.time()

    # Train directly through API
    print("Training started. Streaming logs:")
    print("-" * 50)

    logs = []
    try:
        model = YOLO('yolov8n.pt')
        # Ultralytics автоматически печатает шаги обучения в stdout
        # Если нужны логи, можно временно перенаправить sys.stdout (но редко требуется)
        results = model.train(**train_params)
        end_time = time.time()
        print("\n" + "-" * 50)
        print("Training completed successfully!")

        # Можно извлечь итоговые метрики через results
        # Или собрать их из history (пример):
        metrics = {
            'epochs': train_params['epochs'],
            'best_fitness': getattr(results, 'best_fitness', None),
            'metrics': getattr(results, 'metrics', None),
            # Допиши нужные поля, которые используешь в своей логике
        }

        # Запись метрик
        log_training_metrics(
            experiment_name=experiment_name,
            model_type="yolov8n",
            start_time=start_time,
            end_time=end_time,
            metrics_data=metrics
        )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Activate virtual environment if needed
    # This assumes the script is run from the project root directory
    venv_path = os.path.join(os.path.dirname(__file__), '..', 'venv')
    if os.path.exists(venv_path):
        # Add the virtual environment's Python packages to the path
        site_packages = os.path.join(venv_path, 'Lib', 'site-packages')
        if site_packages not in sys.path:
            sys.path.insert(0, site_packages)

    train_model()

def run_training():
    """
    Run model training directly (for import and call from main.py)
    """
    train_model()
