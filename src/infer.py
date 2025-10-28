import subprocess
import sys
import os
import time
import argparse
from utils import parse_yolo_output, log_inference_metrics, get_experiment_name

"""
Скрипт для запуска инференса (предсказаний) обученной модели YOLO на новых изображениях

Инференс используется для получения предсказаний модели на новых, ранее не виденных данных.
Основные цели инференса:
- Обнаружение дефектов на новых изображениях печатных плат
- Получение bounding boxes и уверенности предсказаний
- Фокусировка на скорости и эффективности обработки

Отличие от валидации:
- Инференс фокусируется на получении реальных предсказаний для практических задач
- Валидация фокусируется на оценке качества модели с использованием метрик точности

Инференс может выполняться на любых изображениях, включая тестовые, валидационные или 
совершенно новые данные из реальных условий эксплуатации.
"""

def run_inference(model_path=None, source_path=None, output_path="results"):
    """
    Run inference on PCB images using trained YOLOv8 model
    
    Args:
        model_path (str): Path to the trained model file
        source_path (str): Path to the source images directory
        output_path (str): Path to the output directory
    """
    # Define default paths if not provided
    if model_path is None:
        # Try the optimized model first, then fall back to other models
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
            print("No trained model found. Please train the model first.")
            print("Expected model locations:")
            for path in model_paths:
                print(f"  - {path}")
            sys.exit(1)
    
    if source_path is None:
        source_path = "datasets/pcb/images/val"
    
    # Generate experiment name
    experiment_name = get_experiment_name("inference")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        sys.exit(1)
    
    # Check if source directory exists
    if not os.path.exists(source_path):
        print(f"Source directory {source_path} not found. Please check your dataset.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Define the command for inference
    cmd = [
        'yolo',
        'predict',
        f'model={model_path}',
        f'source={source_path}',
        f'project={output_path}',
        'save=True',  # Save images with bounding boxes
        'save_txt=True',  # Save detection results in text files
        'save_conf=True',  # Save confidence scores
        'show_labels=True',  # Show class labels on bounding boxes
        'exist_ok=True',   # Overwrite existing results
        'conf=0.25'  # Confidence threshold
    ]
    
    print(f"Running inference with model: {model_path}")
    print(f"Source images: {source_path}")
    print(f"Output directory: {output_path}")
    print("-" * 50)
    
    # Start time measurement
    start_time = time.time()
    
    # Collect output lines for metrics parsing
    output_lines = []
    
    # Execute the command with real-time output streaming
    try:
        # Use Popen to stream output in real-time with proper encoding
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace'  # Replace characters that can't be decoded
        )
        
        # Stream output in real-time
        print("Inference started. Streaming logs:")
        print("-" * 50)
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                output_lines.append(output.strip())
                sys.stdout.flush()
        
        # Wait for process to complete
        process.wait()
        
        # End time measurement
        end_time = time.time()
        
        if process.returncode == 0:
            print("\n" + "-" * 50)
            print("Inference completed successfully!")
            print(f"Results saved to: {output_path}")
            print("Images with defect bounding boxes are saved in the predict directory.")
            
            # Parse metrics from output
            metrics = parse_yolo_output(output_lines)
            
            # Log inference metrics
            log_inference_metrics(
                experiment_name=experiment_name,
                start_time=start_time,
                end_time=end_time,
                metrics_data=metrics
            )
        else:
            print(f"\nInference failed with return code: {process.returncode}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nInference interrupted by user.")
        process.terminate()
        sys.exit(1)
    except Exception as e:
        print(f"Inference failed with error: {e}")
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
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run inference on PCB images using trained YOLOv8 model")
    parser.add_argument("--model", type=str, help="Path to the trained model file")
    parser.add_argument("--source", type=str, help="Path to the source images directory")
    parser.add_argument("--output", type=str, default="results", help="Path to the output directory")
    
    args = parser.parse_args()
    
    run_inference(model_path=args.model, source_path=args.source, output_path=args.output)

# Function for direct import and call
def run_inference_direct(output_path="results"):
    """
    Run inference directly (for import and call from main.py)
    
    Args:
        output_path (str): Path to the output directory for inference results
    """
    run_inference(output_path=output_path)
