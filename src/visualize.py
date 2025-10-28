import os
import sys
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import numpy as np

def visualize_training_metrics():
    """
    Visualize training metrics (precision, recall, mAP) from YOLOv8 training results
    """
    # Define paths
    results_path = "runs/detect/train/results.csv"
    output_dir = "reports/metrics"
    
    # Check if results file exists
    if not os.path.exists(results_path):
        print(f"Results file {results_path} not found. Please train the model first.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read results
    try:
        # Read CSV file with pandas
        results = pd.read_csv(results_path)
    except ImportError:
        print("pandas library is required for visualization. Please install it with: pip install pandas")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading results file: {e}")
        sys.exit(1)
    
    # Extract metrics
    epochs = results['epoch']
    
    # Handle potential variations in column names
    precision_col = None
    recall_col = None
    map50_col = None
    map5095_col = None
    
    # Also look for speed metrics
    speed_preprocess_col = None
    speed_inference_col = None
    speed_postprocess_col = None
    
    for col in results.columns:
        if 'precision' in col.lower():
            precision_col = col
        elif 'recall' in col.lower():
            recall_col = col
        elif 'map50' in col.lower() and '95' not in col.lower():
            map50_col = col
        elif 'map50' in col.lower() and '95' in col.lower():
            map5095_col = col
        elif 'preprocess' in col.lower() and 'speed' in col.lower():
            speed_preprocess_col = col
        elif 'inference' in col.lower() and 'speed' in col.lower():
            speed_inference_col = col
        elif 'postprocess' in col.lower() and 'speed' in col.lower():
            speed_postprocess_col = col
    
    # Plot metrics
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('YOLOv8 Training Metrics', fontsize=16)
    
    # Precision
    if precision_col:
        axes[0, 0].plot(epochs, results[precision_col], label=precision_col, color='blue')
        axes[0, 0].set_title('Precision')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].grid(True)
    
    # Recall
    if recall_col:
        axes[0, 1].plot(epochs, results[recall_col], label=recall_col, color='green')
        axes[0, 1].set_title('Recall')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].grid(True)
    
    # mAP@0.5
    if map50_col:
        axes[0, 2].plot(epochs, results[map50_col], label=map50_col, color='red')
        axes[0, 2].set_title('mAP@0.5')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('mAP@0.5')
        axes[0, 2].grid(True)
    
    # mAP@0.5:0.95
    if map5095_col:
        axes[1, 0].plot(epochs, results[map5095_col], label=map5095_col, color='purple')
        axes[1, 0].set_title('mAP@0.5:0.95')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mAP@0.5:0.95')
        axes[1, 0].grid(True)
    
    # Speed metrics
    if speed_preprocess_col or speed_inference_col or speed_postprocess_col:
        if speed_preprocess_col:
            axes[1, 1].plot(epochs, results[speed_preprocess_col], label='Preprocess', color='orange')
        if speed_inference_col:
            axes[1, 1].plot(epochs, results[speed_inference_col], label='Inference', color='cyan')
        if speed_postprocess_col:
            axes[1, 1].plot(epochs, results[speed_postprocess_col], label='Postprocess', color='magenta')
        axes[1, 1].set_title('Processing Speed')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (ms)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    else:
        axes[1, 1].set_visible(False)
    
    # Combined accuracy metrics
    axes[1, 2].set_title('Combined Accuracy Metrics')
    if precision_col:
        axes[1, 2].plot(epochs, results[precision_col], label='Precision', color='blue')
    if recall_col:
        axes[1, 2].plot(epochs, results[recall_col], label='Recall', color='green')
    if map50_col:
        axes[1, 2].plot(epochs, results[map50_col], label='mAP@0.5', color='red')
    if map5095_col:
        axes[1, 2].plot(epochs, results[map5095_col], label='mAP@0.5:0.95', color='purple')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Value')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_metrics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training metrics visualization saved to {plot_path}")

def visualize_experiment_comparison(metrics_file="metrics/summary.csv"):
    """
    Visualize comparison of different experiments
    
    Args:
        metrics_file (str): Path to the metrics summary file
    """
    # Check if metrics file exists
    if not os.path.exists(metrics_file):
        print(f"Metrics file {metrics_file} not found.")
        print("Please run training, validation, or inference first to generate metrics.")
        return
    
    # Create output directory
    output_dir = "reports/comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # Read metrics data
    try:
        df = pd.read_csv(metrics_file)
    except Exception as e:
        print(f"Error reading metrics file: {e}")
        return
    
    if df.empty:
        print("Metrics file is empty.")
        return
    
    # Filter out rows with no data
    df = df.dropna(how='all')
    
    if df.empty:
        print("No valid experiment data found.")
        return
    
    # Convert time columns to numeric
    time_columns = ['training_time', 'validation_time', 'inference_time']
    for col in time_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert accuracy columns to numeric
    accuracy_columns = [
        'training_precision', 'training_recall', 'training_mAP50', 'training_mAP50-95',
        'validation_precision', 'validation_recall', 'validation_mAP50', 'validation_mAP50-95',
        'inference_precision', 'inference_recall', 'inference_mAP50', 'inference_mAP50-95'
    ]
    for col in accuracy_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create visualizations
    
    # 1. Training time comparison
    if 'training_time' in df.columns and not df['training_time'].isna().all():
        plt.figure(figsize=(12, 6))
        experiments = df['experiment_name'].tolist()
        times = df['training_time'].tolist()
        
        # Filter out NaN values
        valid_data = [(exp, time) for exp, time in zip(experiments, times) if not pd.isna(time)]
        if valid_data:
            experiments, times = zip(*valid_data)
            
            bars = plt.bar(range(len(experiments)), times, color='skyblue')
            plt.xlabel('Experiments')
            plt.ylabel('Training Time (seconds)')
            plt.title('Training Time Comparison')
            plt.xticks(range(len(experiments)), experiments, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, time in zip(bars, times):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                        f'{time:.1f}s', ha='center', va='bottom')
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, "training_time_comparison.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Training time comparison saved to {plot_path}")
    
    # 2. Accuracy comparison (mAP50)
    plt.figure(figsize=(15, 8))
    
    # Collect accuracy data for each stage
    stages = ['training', 'validation', 'inference']
    stage_data = {}
    
    for stage in stages:
        map50_col = f'{stage}_mAP50'
        if map50_col in df.columns:
            # Filter out NaN values
            valid_data = df[['experiment_name', map50_col]].dropna()
            if not valid_data.empty:
                stage_data[stage] = valid_data
    
    if stage_data:
        n_stages = len(stage_data)
        n_experiments = len(df)
        
        # Create subplot for each stage
        fig, axes = plt.subplots(1, n_stages, figsize=(6*n_stages, 6))
        if n_stages == 1:
            axes = [axes]
        
        for i, (stage, data) in enumerate(stage_data.items()):
            bars = axes[i].bar(range(len(data)), data[data.columns[1]], color='lightcoral')
            axes[i].set_xlabel('Experiments')
            axes[i].set_ylabel('mAP@0.5')
            axes[i].set_title(f'{stage.capitalize()} mAP@0.5 Comparison')
            axes[i].set_xticks(range(len(data)))
            axes[i].set_xticklabels(data['experiment_name'], rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, data[data.columns[1]]):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                            f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "map50_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"mAP@0.5 comparison saved to {plot_path}")
    
    # 3. Combined performance radar chart (if we have enough data)
    # This will show a radar chart for the latest experiment with all available metrics
    if len(df) > 0:
        latest_exp = df.iloc[-1]  # Get the latest experiment
        
        # Collect available metrics
        metrics = []
        values = []
        labels = []
        
        # Accuracy metrics
        for metric in ['training_mAP50', 'validation_mAP50', 'inference_mAP50']:
            if metric in df.columns and not pd.isna(latest_exp[metric]):
                metrics.append(metric)
                values.append(float(latest_exp[metric]))
                labels.append(metric.replace('_mAP50', '').capitalize())
        
        # If we have accuracy metrics, create radar chart
        if len(values) > 0:
            # Normalize values for radar chart (0-1 scale)
            normalized_values = [(v - min(values)) / (max(values) - min(values)) if max(values) != min(values) else 0.5 for v in values]
            # Repeat the first value to close the circle
            normalized_values += normalized_values[:1]
            
            # Calculate angles for radar chart
            angles = [n / float(len(labels)) * 2 * np.pi for n in range(len(labels))]
            angles += angles[:1]
            
            # Create radar chart
            plt.figure(figsize=(8, 8))
            ax = plt.subplot(111, polar=True)
            
            # Draw one axe per variable + add labels
            plt.xticks(angles[:-1], labels, color='grey', size=12)
            
            # Draw ylabels
            ax.set_rlabel_position(0)
            plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=10)
            plt.ylim(0, 1)
            
            # Plot data
            ax.plot(angles, normalized_values, linewidth=2, linestyle='solid', label='Performance')
            ax.fill(angles, normalized_values, 'b', alpha=0.1)
            
            # Add values as text
            for angle, value, orig_value in zip(angles[:-1], normalized_values[:-1], values):
                ax.text(angle, value + 0.1, f'{orig_value:.4f}', 
                       horizontalalignment='center', size=10, color='black')
            
            plt.title(f"Performance Radar Chart - {latest_exp['experiment_name']}", size=16, pad=20)
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            plot_path = os.path.join(output_dir, "performance_radar.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Performance radar chart saved to {plot_path}")
    
    print(f"All comparison visualizations saved to {output_dir}")

def main():
    # Activate virtual environment if needed
    # This assumes the script is run from the project root directory
    venv_path = os.path.join(os.path.dirname(__file__), '..', 'venv')
    if os.path.exists(venv_path):
        # Add the virtual environment's Python packages to the path
        site_packages = os.path.join(venv_path, 'Lib', 'site-packages')
        if site_packages not in sys.path:
            sys.path.insert(0, site_packages)
    
    # Try to import required libraries
    try:
        import matplotlib
        import pandas
        import numpy as np
    except ImportError as e:
        print(f"Required libraries not found: {e}")
        print("Please install required libraries with: pip install matplotlib pandas numpy")
        sys.exit(1)
    
    # Visualize training metrics
    print("Generating training metrics visualization...")
    visualize_training_metrics()
    
    # Visualize experiment comparison
    print("Generating experiment comparison visualizations...")
    visualize_experiment_comparison()
    
    print("All visualizations completed!")

if __name__ == "__main__":
    main()
