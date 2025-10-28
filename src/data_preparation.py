import os
import shutil
import sys
from pathlib import Path

try:
    import kagglehub
except ImportError:
    print("kagglehub library is required for data preparation. Please install it with: pip install kagglehub")
    sys.exit(1)

def download_and_prepare_dataset():
    """
    Download PCB defect dataset from Kaggle and organize it in YOLO format
    """
    # Define paths
    dataset_path = "datasets/pcb"
    images_path = os.path.join(dataset_path, "images")
    labels_path = os.path.join(dataset_path, "labels")
    
    # Create directory structure
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)
    os.makedirs(os.path.join(images_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(images_path, "val"), exist_ok=True)
    os.makedirs(os.path.join(images_path, "test"), exist_ok=True)
    os.makedirs(os.path.join(labels_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(labels_path, "val"), exist_ok=True)
    os.makedirs(os.path.join(labels_path, "test"), exist_ok=True)
    
    print("Downloading PCB defect dataset from Kaggle...")
    
    # Download dataset from Kaggle
    # Dataset: https://www.kaggle.com/datasets/norbertelter/pcb-defect-dataset
    try:
        path = kagglehub.dataset_download("norbertelter/pcb-defect-dataset")
        print(f"Dataset downloaded to: {path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please make sure you have set up your Kaggle credentials correctly.")
        print("Follow instructions at: https://github.com/Kaggle/kagglehub/blob/main/README.md#authenticate")
        path = "./datasets/pcb"
        # sys.exit(1)
    
    # Organize files
    print("Organizing dataset files...")
    
    # Move files to appropriate directories
    # This is a simplified example - you may need to adjust based on the actual dataset structure
    source_images_train = os.path.join(path, "pcb-defect-dataset", "train", "images")
    source_labels_train = os.path.join(path, "pcb-defect-dataset", "train", "labels")
    source_images_val = os.path.join(path, "pcb-defect-dataset", "val", "images")
    source_labels_val = os.path.join(path, "pcb-defect-dataset", "val", "labels")
    source_images_test = os.path.join(path, "pcb-defect-dataset", "test", "images")
    source_labels_test = os.path.join(path, "pcb-defect-dataset", "test", "labels")
    
    # Check if the expected directory structure exists
    if os.path.exists(source_images_train) and os.path.exists(source_labels_train):
        # Move training images
        for file in os.listdir(source_images_train):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy2(
                    os.path.join(source_images_train, file),
                    os.path.join(images_path, "train", file)
                )
        
        # Move training labels
        for file in os.listdir(source_labels_train):
            if file.endswith('.txt'):
                shutil.copy2(
                    os.path.join(source_labels_train, file),
                    os.path.join(labels_path, "train", file)
                )
    
    if os.path.exists(source_images_val) and os.path.exists(source_labels_val):
        # Move validation images
        for file in os.listdir(source_images_val):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy2(
                    os.path.join(source_images_val, file),
                    os.path.join(images_path, "val", file)
                )
        
        # Move validation labels
        for file in os.listdir(source_labels_val):
            if file.endswith('.txt'):
                shutil.copy2(
                    os.path.join(source_labels_val, file),
                    os.path.join(labels_path, "val", file)
                )
                
    if os.path.exists(source_images_test) and os.path.exists(source_labels_test):
        # Move test images
        for file in os.listdir(source_images_test):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy2(
                    os.path.join(source_images_test, file),
                    os.path.join(images_path, "test", file)
                )
        
        # Move test labels
        for file in os.listdir(source_labels_test):
            if file.endswith('.txt'):
                shutil.copy2(
                    os.path.join(source_labels_test, file),
                    os.path.join(labels_path, "test", file)
                )
    
    print("Dataset preparation completed!")
    print(f"Dataset is organized in: {dataset_path}")
    print("Structure:")
    print("  images/")
    print("    train/")
    print("    val/")
    print("    test/")
    print("  labels/")
    print("    train/")
    print("    val/")
    print("    test/")

def main():
    # Activate virtual environment if needed
    # This assumes the script is run from the project root directory
    venv_path = os.path.join(os.path.dirname(__file__), '..', 'venv')
    if os.path.exists(venv_path):
        # Add the virtual environment's Python packages to the path
        site_packages = os.path.join(venv_path, 'Lib', 'site-packages')
        if site_packages not in sys.path:
            sys.path.insert(0, site_packages)
    
    download_and_prepare_dataset()

if __name__ == "__main__":
    main()

# Function for direct import and call
def run_data_preparation():
    """
    Run data preparation directly (for import and call from main.py)
    """
    download_and_prepare_dataset()