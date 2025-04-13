#!/usr/bin/env python3
"""
Visualization script for generating additional plots and metrics for the health data classification project.
This script creates visualizations for feature importance, model performance, and task correlations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.preprocessing import label_binarize

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
                    for target_class in train_df[target_col].unique():
                        subset = train_df[train_df[target_col] == target_class]
                        sns.kdeplot(subset[feature], label=target_class)
                    
                    plt.title(f'{feature} Distribution by {task}')
                    plt.xlabel(feature)
                    plt.ylabel('Density')
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
        
        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=0, vmax=1)
        plt.title('Task Correlations (Cramer\'s V)')
        plt.tight_layout()
        plt.savefig(save_dir / 'task_correlations.png')
        plt.close()
        print("Saved task correlations plot")

def plot_roc_curves(save_dir=VIZ_DIR):
    """Plot ROC curves for binary classification tasks."""
    # This function would typically use model predictions
    # Since we can't run the model due to disk space limitations,
    # we'll provide the code structure for the user to run later
    
    print("""
    # Code to generate ROC curves (to be run after model training)
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    def plot_roc_curves(model, test_loader, label_encoders, device, save_dir):
        model.eval()
        
        # For storing predictions and targets
        all_probs = {task: [] for task in label_encoders.keys()}
        all_targets = {task: [] for task in label_encoders.keys()}
        
        with torch.no_grad():
            for features, targets in test_loader:
                # Move data to device
                features = features.to(device)
                
                # Forward pass
                outputs = model(features)
                
                # Get probabilities
                for task, output in outputs.items():
                    probs = torch.softmax(output, dim=1)
                    all_probs[task].extend(probs.cpu().numpy())
                    all_targets[task].extend(targets[task].cpu().numpy())
        
        # Plot ROC curves for binary tasks
        binary_tasks = ['Microbiome Status', 'Digestive Health']
        
        for task in binary_tasks:
            if task in all_probs:
                y_true = np.array(all_targets[task])
                y_score = np.array(all_probs[task])[:, 1]  # Probability of positive class
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                
                # Plot
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {task}')
                plt.legend(loc='lower right')
                plt.savefig(save_dir / f'roc_curve_{task}.png')
                plt.close()
    """)
    
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
    # This function would typically use model predictions
    # Since we can't run the model due to disk space limitations,
    # we'll provide the code structure for the user to run later
    
    print("""
    # Code to generate precision-recall curves (to be run after model training)
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    def plot_precision_recall_curves(model, test_loader, label_encoders, device, save_dir):
        model.eval()
        
        # For storing predictions and targets
        all_probs = {task: [] for task in label_encoders.keys()}
        all_targets = {task: [] for task in label_encoders.keys()}
        
        with torch.no_grad():
            for features, targets in test_loader:
                # Move data to device
                features = features.to(device)
                
                # Forward pass
                outputs = model(features)
                
                # Get probabilities
                for task, output in outputs.items():
                    probs = torch.softmax(output, dim=1)
                    all_probs[task].extend(probs.cpu().numpy())
                    all_targets[task].extend(targets[task].cpu().numpy())
        
        # Plot precision-recall curves for each task
        for task in all_probs.keys():
            y_true = np.array(all_targets[task])
            
            # For binary classification
            if len(label_encoders[task]) == 2:
                y_score = np.array(all_probs[task])[:, 1]  # Probability of positive class
                
                # Calculate precision-recall curve
                precision, recall, _ = precision_recall_curve(y_true, y_score)
                ap = average_precision_score(y_true, y_score)
                
                # Plot
                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {ap:.2f})')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.ylim([0.0, 1.05])
                plt.xlim([0.0, 1.0])
                plt.title(f'Precision-Recall Curve - {task}')
                plt.legend(loc='lower left')
                plt.savefig(save_dir / f'pr_curve_{task}.png')
                plt.close()
            
            # For multi-class classification
            else:
                y_true_bin = label_binarize(y_true, classes=range(len(label_encoders[task])))
                y_score = np.array(all_probs[task])
                
                # Plot precision-recall curve for each class
                plt.figure(figsize=(10, 8))
                
                for i in range(len(label_encoders[task])):
                    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
                    ap = average_precision_score(y_true_bin[:, i], y_score[:, i])
                    plt.plot(recall, precision, lw=2, label=f'Class {i} (AP = {ap:.2f})')
                
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.ylim([0.0, 1.05])
                plt.xlim([0.0, 1.0])
                plt.title(f'Precision-Recall Curve - {task}')
                plt.legend(loc='lower left')
                plt.savefig(save_dir / f'pr_curve_{task}.png')
                plt.close()
    """)
    
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
