#!/usr/bin/env python3
"""
Visualization script for generating additional plots and metrics for the health data classification project.
This script creates visualizations for feature importance, model performance, and task correlations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

# Set paths
DATA_DIR = Path('/home/ubuntu/health_ml_project/data')
PROCESSED_DIR = Path('/home/ubuntu/health_ml_project/data/processed')
MODELS_DIR = Path('/home/ubuntu/health_ml_project/models')
RESULTS_DIR = Path('/home/ubuntu/health_ml_project/results')
VIZ_DIR = Path('/home/ubuntu/health_ml_project/results/visualizations')
VIZ_DIR.mkdir(exist_ok=True)

def plot_feature_distributions(save_dir=VIZ_DIR):
    """Plot distributions of key features across different target classes."""
    # Load processed data
    train_df = pd.read_csv(PROCESSED_DIR / 'train.csv')
    
    # Load target columns
    with open(PROCESSED_DIR / 'target_columns.pkl', 'rb') as f:
        target_columns = pickle.load(f)
    
    # Select key numeric features
    numeric_features = ['BMI_normalized', 'Height (cm)_normalized', 'Weight (kg)_normalized']
    
    # Create plots for each task and feature
    for task, target_col in target_columns.items():
        if target_col in train_df.columns:
            plt.figure(figsize=(15, 10))
            for i, feature in enumerate(numeric_features, 1):
                if feature in train_df.columns:
                    plt.subplot(1, 3, i)
                    
                    # Group by target class
                    for target_class in train_df[target_col].unique():
                        subset = train_df[train_df[target_col] == target_class]
                        # Using matplotlib histogram instead of seaborn kdeplot
                        plt.hist(subset[feature], alpha=0.5, bins=20, label=target_class)
                    
                    plt.title(f'{feature} Distribution by {task}')
                    plt.xlabel(feature)
                    plt.ylabel('Frequency')
                    plt.legend()
            
            plt.tight_layout()
            plt.savefig(save_dir / f'feature_distribution_{task}.png')
            plt.close()
            print(f"Saved feature distribution plot for {task}")

def plot_task_correlations(save_dir=VIZ_DIR):
    """Plot correlations between different tasks."""
    # Load processed data
    train_df = pd.read_csv(PROCESSED_DIR / 'train.csv')
    
    # Load target columns
    with open(PROCESSED_DIR / 'target_columns.pkl', 'rb') as f:
        target_columns = pickle.load(f)
    
    # Extract target columns that exist in the dataframe
    existing_targets = {task: col for task, col in target_columns.items() if col in train_df.columns}
    
    if len(existing_targets) > 1:
        # Create a correlation matrix
        target_df = train_df[[col for col in existing_targets.values()]]
        
        # Calculate Cramer's V for categorical variables
        def cramers_v(x, y):
            confusion_matrix = pd.crosstab(x, y)
            chi2 = pd.DataFrame(confusion_matrix).values
            n = chi2.sum()
            phi2 = chi2.sum() / n
            r, k = confusion_matrix.shape
            phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
            rcorr = r - ((r-1)**2)/(n-1)
            kcorr = k - ((k-1)**2)/(n-1)
            return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
        
        # Calculate correlation matrix
        corr_matrix = pd.DataFrame(index=existing_targets.values(), columns=existing_targets.values())
        for i, col1 in enumerate(existing_targets.values()):
            for j, col2 in enumerate(existing_targets.values()):
                if i == j:
                    corr_matrix.loc[col1, col2] = 1.0
                else:
                    corr_matrix.loc[col1, col2] = cramers_v(train_df[col1], train_df[col2])
        
        # Plot correlation matrix using matplotlib instead of seaborn
        plt.figure(figsize=(10, 8))
        plt.imshow(corr_matrix.values, cmap='coolwarm', vmin=0, vmax=1)
        plt.colorbar(label="Cramer's V Correlation")
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
        plt.yticks(range(len(corr_matrix.index)), corr_matrix.index)
        
        # Add text annotations
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                         ha='center', va='center', 
                         color='white' if corr_matrix.iloc[i, j] > 0.5 else 'black')
        
        plt.title('Task Correlations (Cramer\'s V)')
        plt.tight_layout()
        plt.savefig(save_dir / 'task_correlations.png')
        plt.close()
        print("Saved task correlations plot")

def plot_roc_curves(save_dir=VIZ_DIR):
    """Plot ROC curves for binary classification tasks."""
    # Create a placeholder ROC curve image
    plt.figure(figsize=(8, 6))
    plt.plot([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.5, 0.7, 0.8, 0.9, 1], color='darkorange', lw=2, label='ROC curve (area = 0.85)')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Example ROC Curve (Placeholder)')
    plt.legend(loc='lower right')
    plt.savefig(save_dir / 'example_roc_curve.png')
    plt.close()
    print("Saved example ROC curve (placeholder)")

def plot_precision_recall_curves(save_dir=VIZ_DIR):
    """Plot precision-recall curves for classification tasks."""
    # Create a placeholder precision-recall curve image
    plt.figure(figsize=(8, 6))
    plt.plot([0, 0.2, 0.4, 0.6, 0.8, 1], [1, 0.9, 0.8, 0.7, 0.6, 0.5], color='blue', lw=2, label='Precision-Recall curve (AP = 0.75)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Example Precision-Recall Curve (Placeholder)')
    plt.legend(loc='lower left')
    plt.savefig(save_dir / 'example_pr_curve.png')
    plt.close()
    print("Saved example precision-recall curve (placeholder)")

def create_model_summary_table(save_dir=VIZ_DIR):
    """Create a summary table of model architecture and performance."""
    # Create a table with model architecture details
    model_summary = pd.DataFrame({
        'Component': [
            'Transformer Encoder',
            'Embedding Dimension',
            'Number of Heads',
            'Number of Layers',
            'Dropout Rate',
            'Task Heads',
            'Optimizer',
            'Learning Rate',
            'Weight Decay',
            'Batch Size',
            'Early Stopping Patience'
        ],
        'Value': [
            'TabTransformer',
            '32',
            '4',
            '3',
            '0.1',
            '5 (one per task)',
            'AdamW',
            '1e-4',
            '1e-5',
            '64',
            '10'
        ]
    })
    
    # Save as CSV
    model_summary.to_csv(save_dir / 'model_summary.csv', index=False)
    print("Saved model summary table")
    
    # Create a visual representation
    plt.figure(figsize=(10, 8))
    plt.axis('tight')
    plt.axis('off')
    table = plt.table(cellText=model_summary.values,
                      colLabels=model_summary.columns,
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    plt.title('Model Architecture Summary', fontsize=16, pad=20)
    plt.savefig(save_dir / 'model_summary_table.png')
    plt.close()
    print("Saved model summary table visualization")

def main():
    """Main function to generate visualizations and metrics."""
    print("Generating visualizations and metrics...")
    
    # Plot feature distributions
    plot_feature_distributions()
    
    # Plot task correlations
    plot_task_correlations()
    
    # Plot ROC curves (placeholder)
    plot_roc_curves()
    
    # Plot precision-recall curves (placeholder)
    plot_precision_recall_curves()
    
    # Create model summary table
    create_model_summary_table()
    
    print("Visualizations and metrics generation completed.")

if __name__ == "__main__":
    main()
