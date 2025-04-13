#!/usr/bin/env python3
"""
Utility script to demonstrate sample predictions from the multi-task learning model.
This script shows how to make predictions for new data points and interpret the results.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Import model and dataset classes
from model import MultiTaskModel, create_model

# Set paths
DATA_DIR = Path('/home/ubuntu/health_ml_project/data')
PROCESSED_DIR = Path('/home/ubuntu/health_ml_project/data/processed')
MODELS_DIR = Path('/home/ubuntu/health_ml_project/models')
RESULTS_DIR = Path('/home/ubuntu/health_ml_project/results')

def load_model_and_encoders():
    """Load the trained model and label encoders."""
    # Load label encoders
    with open(MODELS_DIR / 'label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    # Get number of classes for each task
    num_classes = {task: len(encoder) for task, encoder in label_encoders.items()}
    
    # Load test data to get feature dimension
    test_df = pd.read_csv(PROCESSED_DIR / 'test.csv')
    
    # Load target columns
    with open(PROCESSED_DIR / 'target_columns.pkl', 'rb') as f:
        target_columns = pickle.load(f)
    
    # Get feature columns (all columns except target columns)
    feature_cols = [col for col in test_df.columns if col not in target_columns.values()]
    feature_dim = len(feature_cols)
    
    # Create model
    model = create_model(feature_dim, num_classes)
    
    # Load trained weights
    model.load_state_dict(torch.load(MODELS_DIR / 'best_model.pt', map_location=torch.device('cpu')))
    model.eval()
    
    return model, label_encoders, feature_cols, target_columns

def make_predictions(model, data, feature_cols, label_encoders):
    """Make predictions for new data points."""
    # Extract features
    features = torch.tensor(data[feature_cols].values, dtype=torch.float32)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(features)
    
    # Process predictions
    predictions = {}
    probabilities = {}
    
    for task, output in outputs.items():
        # Get predicted class indices
        _, predicted_indices = torch.max(output, 1)
        predicted_indices = predicted_indices.numpy()
        
        # Get class probabilities
        probs = torch.softmax(output, dim=1).numpy()
        
        # Convert indices to class names
        idx_to_class = {idx: cls for cls, idx in label_encoders[task].items()}
        predicted_classes = [idx_to_class[idx] for idx in predicted_indices]
        
        predictions[task] = predicted_classes
        probabilities[task] = probs
    
    return predictions, probabilities

def plot_sample_predictions(test_df, predictions, probabilities, label_encoders, target_columns, num_samples=5):
    """Plot sample predictions with true vs predicted classes."""
    # Select random samples
    sample_indices = np.random.choice(len(test_df), num_samples, replace=False)
    samples = test_df.iloc[sample_indices]
    
    # Create figure
    fig, axes = plt.subplots(len(target_columns), 1, figsize=(12, 4 * len(target_columns)))
    
    for i, (task, col) in enumerate(target_columns.items()):
        ax = axes[i]
        
        # Get true and predicted classes for samples
        true_classes = samples[col].values
        pred_classes = [predictions[task][j] for j in sample_indices]
        
        # Get probabilities for samples
        probs = [probabilities[task][j] for j in sample_indices]
        
        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'Sample': [f'Sample {j+1}' for j in range(num_samples)],
            'True Class': true_classes,
            'Predicted Class': pred_classes
        })
        
        # Plot
        sns.barplot(x='Sample', y='True Class', data=plot_data, ax=ax, label='True', color='blue', alpha=0.5)
        sns.barplot(x='Sample', y='Predicted Class', data=plot_data, ax=ax, label='Predicted', color='red', alpha=0.5)
        
        ax.set_title(f'Task: {task}')
        ax.legend()
        
        # Add probability annotations
        for j in range(num_samples):
            # Get class names and probabilities
            idx_to_class = {idx: cls for cls, idx in label_encoders[task].items()}
            class_names = [idx_to_class[idx] for idx in range(len(label_encoders[task]))]
            sample_probs = probs[j]
            
            # Add text annotations
            for k, (cls, prob) in enumerate(zip(class_names, sample_probs)):
                ax.text(j, k * 0.1, f'{cls}: {prob:.2f}', fontsize=8, ha='center')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'sample_predictions.png')
    plt.close()

def export_results_to_pdf():
    """Export results (plots and metrics) to a PDF file."""
    # This function would typically use a library like reportlab or matplotlib's PdfPages
    # Since we're providing code for the user to run elsewhere, we'll just provide instructions
    
    print("To export results to PDF:")
    print("1. Install required libraries: pip install reportlab matplotlib")
    print("2. Use the following code to create a PDF report:")
    print("""
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    import pickle
    from pathlib import Path
    
    # Set paths
    RESULTS_DIR = Path('/path/to/results')
    
    # Create PDF
    with PdfPages(RESULTS_DIR / 'health_data_classification_report.pdf') as pdf:
        # Add learning curves
        plt.figure(figsize=(12, 8))
        img = plt.imread(RESULTS_DIR / 'learning_curves.png')
        plt.imshow(img)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # Add confusion matrices
        for task in ['Microbiome Status', 'Obesity Risk', 'Digestive Health', 
                    'Chronic Condition Risk', 'Supplement Recommendation']:
            plt.figure(figsize=(12, 8))
            img = plt.imread(RESULTS_DIR / f'confusion_matrix_{task}.png')
            plt.imshow(img)
            plt.axis('off')
            pdf.savefig()
            plt.close()
        
        # Add sample predictions
        plt.figure(figsize=(12, 8))
        img = plt.imread(RESULTS_DIR / 'sample_predictions.png')
        plt.imshow(img)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # Add metrics
        with open(RESULTS_DIR / 'test_metrics.pkl', 'rb') as f:
            test_metrics = pickle.load(f)
        
        plt.figure(figsize=(12, 8))
        plt.text(0.1, 0.9, 'Test Metrics', fontsize=20)
        y_pos = 0.8
        for task, metrics in test_metrics.items():
            plt.text(0.1, y_pos, f'{task}:', fontsize=16)
            y_pos -= 0.05
            for metric_name, metric_value in metrics.items():
                plt.text(0.2, y_pos, f'{metric_name}: {metric_value:.4f}', fontsize=12)
                y_pos -= 0.03
            y_pos -= 0.05
        
        plt.axis('off')
        pdf.savefig()
        plt.close()
    
    print(f"Report saved to {RESULTS_DIR / 'health_data_classification_report.pdf'}")
    """)

def main():
    """Main function to demonstrate sample predictions."""
    try:
        # Load model and encoders
        model, label_encoders, feature_cols, target_columns = load_model_and_encoders()
        
        # Load test data
        test_df = pd.read_csv(PROCESSED_DIR / 'test.csv')
        
        # Make predictions
        predictions, probabilities = make_predictions(model, test_df, feature_cols, label_encoders)
        
        # Plot sample predictions
        plot_sample_predictions(test_df, predictions, probabilities, label_encoders, target_columns)
        
        # Print instructions for exporting results to PDF
        export_results_to_pdf()
        
        print("Sample predictions generated and saved to results directory.")
    except Exception as e:
        print(f"Error: {e}")
        print("This script requires a trained model. Please run train.py first.")
        print("Since PyTorch couldn't be installed due to disk space limitations, you'll need to run these scripts in an environment with sufficient resources.")

if __name__ == "__main__":
    main()
