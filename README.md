# PCB Defect Detection Project using YOLOv8

This project uses YOLOv8 to detect defects on printed circuit boards (PCBs). It includes scripts for data preparation, model training, inference, and visualization.

## Project Structure

- `src/`: Source code directory
  - `data_preparation.py`: Script to download and organize the dataset
  - `train_model.py`: Script to train the YOLOv8 model
  - `infer.py`: Script to run inference on new images
  - `validate_model.py`: Script to validate the trained model accuracy
  - `diagnose_model.py`: Script to diagnose model and data issues
  - `visualize.py`: Script to visualize results
  - `compare_experiments.py`: Script to compare metrics of different experiments
  - `main.py`: Main script to automate the entire pipeline (moved to project root)
  - `utils.py`: Utility functions for metrics logging and comparison
- `datasets/`: Directory for storing datasets
- `chip_defects.yaml`: Dataset configuration file
- `requirements.txt`: Python dependencies
- `metrics/`: Directory for storing performance metrics of different experiments

## Difference Between Validation and Inference

### Validation
Validation is the process of evaluating a trained model's performance on a separate dataset (validation set) that was not used during training. The main purposes of validation are:
- Assessing model accuracy using metrics like precision, recall, and mAP
- Checking for overfitting by comparing training and validation performance
- Tuning hyperparameters based on validation results
- Providing quantitative measures of model quality

The `validate_model.py` script performs validation and provides detailed accuracy metrics.

### Inference
Inference is the process of using a trained model to make predictions on new, unseen data. The main purposes of inference are:
- Detecting defects on new PCB images
- Providing real-world predictions for practical applications
- Focusing on prediction speed and efficiency
- Generating bounding boxes and confidence scores for detected defects

The `infer.py` script performs inference and saves detection results.

## How to Use

### 1. Environment Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### 2. Automated Pipeline

For a complete automated workflow, use the main script:
```
python main.py
```

This will run all steps in sequence:
1. Data preparation
2. Model training
3. Model validation
4. Inference
5. Metrics collection
6. Visualization

All steps are now called directly through function calls rather than subprocess execution for better performance and error handling.

You can skip specific steps using the `--skip` argument:
```
python main.py --skip training validation
```

### 3. Data Preparation

1. Set up your Kaggle credentials by following the instructions at: https://github.com/Kaggle/kagglehub/blob/main/README.md#authenticate

2. Run the data preparation script:
   ```
   python src/data_preparation.py
   ```

This script will:
- Download the PCB defect dataset from Kaggle
- Organize the dataset in YOLO format in the `datasets/pcb` directory

### 4. Model Training

1. Run the training script:
   ```
   python src/train_model.py
   ```

This script will:
- Automatically detect and use GPU if available
- Train a YOLOv8 model on the prepared dataset with optimized parameters:
  - Mixed precision training for faster computation
  - Adam optimizer with tuned learning rate
  - Early stopping to prevent overfitting
  - Increased number of workers for data loading
- Save the trained model weights
- Log performance metrics (time, speed, accuracy) for experiment comparison

### 5. Speeding Up Training

To speed up the training process, consider the following optimizations:

1. **Use GPU**: The script automatically detects and uses CUDA-enabled GPU if available.

2. **Increase batch size**: The current batch size is set to 32. If your GPU memory allows, you can increase it to 64 or higher in the `src/train_model.py` file.

3. **Adjust image size**: The default image size is 640. For faster training, you can reduce it to 416 or 320 in `src/train_model.py`.

4. **Use mixed precision**: The script already uses automatic mixed precision (AMP) which reduces memory usage and speeds up training.

5. **Early stopping**: The script implements early stopping with a patience of 10 epochs to prevent unnecessary training time.

6. **Optimize data loading**: The script uses 8 worker threads for data loading. Adjust this number based on your CPU capabilities.

7. **Use CUDA for GPU acceleration**: The script automatically detects and uses CUDA if available. To enable GPU acceleration:
   - Install CUDA-compatible PyTorch version:
     ```
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```
   - Install CUDA toolkit from NVIDIA website
   - Verify CUDA installation by running:
     ```python
     import torch
     print(torch.cuda.is_available())
     ```

### 6. Model Validation

1. Run the validation script:
   ```
   python src/validate_model.py
   ```

This script will:
- Validate the trained model accuracy on validation dataset
- Calculate and display accuracy metrics (precision, recall, mAP)
- Log performance metrics for experiment comparison

### 7. Inference

1. Run the inference script:
   ```
   python src/infer.py
   ```

This script will:
- Use the trained model to detect defects in new images
- Save detection results with bounding boxes visualized on images
- Save text files with bounding box coordinates and confidence scores
- Log performance metrics for experiment comparison

Results are saved in the `results/predict` directory:
- Images with visualized defect bounding boxes
- Text files with detection coordinates (`image_name.txt`)
- Confidence scores for each detection

### 8. Experiment Comparison

To compare performance metrics of different experiments:

```
python src/compare_experiments.py
```

This script will:
- Display a table with metrics of all experiments
- Show training time, validation time, inference time
- Show accuracy metrics (mAP50, mAP50-95) for comparison

### 9. Visualization

1. Run the visualization script:
   ```
   python src/visualize.py
   ```

This script will:
- Visualize the training results and inference outputs
- Generate comparison charts for different experiments
- Create performance radar charts

## Troubleshooting

### Dataset path error

If you encounter an error like:
```
Dataset 'chip_defects.yaml' images not found , missing path '.../datasets/datasets/pcb/images/val'
```

This means that the path in `chip_defects.yaml` is incorrectly configured. Make sure the `path` value in `chip_defects.yaml` is set to `datasets/pcb` without a leading dot.

## Dataset

The project uses the "PCB defect dataset" from Kaggle: https://www.kaggle.com/datasets/norbertelter/pcb-defect-dataset

The dataset contains images of PCBs with the following defect types:
- open_circuit
- short
- mouse_bite
- spur
- missing_hole
- spurious_copper