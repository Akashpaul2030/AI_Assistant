#!/usr/bin/env python3
"""
Training script for the multi-task learning model.
This script defines the training pipeline, loss functions, and evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import time
import os
from tqdm import tqdm
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, cohen_kappa_score, 
    precision_score, recall_score, roc_auc_score, 
    average_precision_score, confusion_matrix
)

# Import model and dataset classes
from model import MultiTaskModel, HealthDataset, create_dataloaders, create_model

# Set paths for Windows environment
CURRENT_DIR = Path(os.getcwd())
DATA_DIR = CURRENT_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = CURRENT_DIR / 'models'
RESULTS_DIR = CURRENT_DIR / 'results'

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Print directory information (optional, helpful for debugging)
print(f"Current working directory: {CURRENT_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Processed data directory: {PROCESSED_DIR}")
print(f"Models directory: {MODELS_DIR}")
print(f"Results directory: {RESULTS_DIR}")

# Define constants
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
    
    def __call__(self, val_loss, model):
        """
        Call method to be called at the end of every validation phase.
        
        Args:
            val_loss (float): Validation loss.
            model (nn.Module): Model to save.
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    task_losses = {task: 0 for task in criterion.keys()}
    task_correct = {task: 0 for task in criterion.keys()}
    task_total = {task: 0 for task in criterion.keys()}
    
    for features, targets in tqdm(train_loader, desc="Training"):
        # Move data to device
        features = features.to(device)
        targets = {task: target.to(device) for task, target in targets.items()}
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features)
        
        # Calculate loss for each task
        loss = 0
        for task, output in outputs.items():
            task_loss = criterion[task](output, targets[task])
            loss += task_loss
            task_losses[task] += task_loss.item() * features.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            task_correct[task] += (predicted == targets[task]).sum().item()
            task_total[task] += targets[task].size(0)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * features.size(0)
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(train_loader.dataset)
    task_avg_losses = {task: loss / len(train_loader.dataset) for task, loss in task_losses.items()}
    task_accuracies = {task: correct / total for task, (correct, total) in 
                      zip(task_correct.keys(), zip(task_correct.values(), task_total.values()))}
    
    return avg_loss, task_avg_losses, task_accuracies

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    task_losses = {task: 0 for task in criterion.keys()}
    task_correct = {task: 0 for task in criterion.keys()}
    task_total = {task: 0 for task in criterion.keys()}
    
    # For storing predictions and targets for metrics calculation
    all_predictions = {task: [] for task in criterion.keys()}
    all_targets = {task: [] for task in criterion.keys()}
    all_probs = {task: [] for task in criterion.keys()}
    
    with torch.no_grad():
        for features, targets in tqdm(val_loader, desc="Validation"):
            # Move data to device
            features = features.to(device)
            targets = {task: target.to(device) for task, target in targets.items()}
            
            # Forward pass
            outputs = model(features)
            
            # Calculate loss for each task
            loss = 0
            for task, output in outputs.items():
                task_loss = criterion[task](output, targets[task])
                loss += task_loss
                task_losses[task] += task_loss.item() * features.size(0)
                
                # Calculate accuracy
                _, predicted = torch.max(output, 1)
                task_correct[task] += (predicted == targets[task]).sum().item()
                task_total[task] += targets[task].size(0)
                
                # Store predictions and targets for metrics calculation
                all_predictions[task].extend(predicted.cpu().numpy())
                all_targets[task].extend(targets[task].cpu().numpy())
                all_probs[task].extend(torch.softmax(output, dim=1).cpu().numpy())
            
            total_loss += loss.item() * features.size(0)
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(val_loader.dataset)
    task_avg_losses = {task: loss / len(val_loader.dataset) for task, loss in task_losses.items()}
    task_accuracies = {task: correct / total for task, (correct, total) in 
                      zip(task_correct.keys(), zip(task_correct.values(), task_total.values()))}
    
    # Calculate additional metrics for each task
    task_metrics = {}
    for task in criterion.keys():
        y_true = np.array(all_targets[task])
        y_pred = np.array(all_predictions[task])
        y_probs = np.array(all_probs[task])
        
        metrics = {}
        
        # Task-specific primary and secondary metrics
        if task == 'Microbiome Status':
            # Balanced Accuracy + F1, Cohen's Kappa
            metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
            metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        elif task == 'Obesity Risk':
            # Weighted F1 + Precision/Recall per class
            metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
            try:
                # Added error handling for potential class mismatch issues
                metrics['precision'] = precision_score(y_true, y_pred, average=None, zero_division=0)
                metrics['recall'] = recall_score(y_true, y_pred, average=None, zero_division=0)
            except Exception as e:
                print(f"Error calculating precision/recall for Obesity Risk: {e}")
                metrics['precision'] = 0
                metrics['recall'] = 0
        
        elif task == 'Digestive Health':
            # Macro F1 + AUC-ROC (One-vs-Rest)
            metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
            try:
                # More robust error handling for ROC AUC calculation
                if len(np.unique(y_true)) > 1:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_probs, multi_class='ovr')
                else:
                    print(f"Warning: Only one class present in {task} validation set")
                    metrics['auc_roc'] = 0
            except Exception as e:
                print(f"Error calculating ROC AUC for Digestive Health: {e}")
                metrics['auc_roc'] = 0
        
        elif task == 'Chronic Condition Risk':
            # AUC-PR + Sensitivity at 90% specificity (approximated)
            try:
                # Improved robustness for AUC-PR calculation
                if len(np.unique(y_true)) > 1:
                    if y_probs.shape[1] > 1:
                        # For multi-class, use the probability of the positive class
                        metrics['auc_pr'] = average_precision_score(y_true, y_probs[:, 1])
                    else:
                        metrics['auc_pr'] = average_precision_score(y_true, y_probs)
                else:
                    print(f"Warning: Only one class present in {task} validation set")
                    metrics['auc_pr'] = 0
            except Exception as e:
                print(f"Error calculating PR AUC for Chronic Condition Risk: {e}")
                metrics['auc_pr'] = 0
            
            # Sensitivity calculation with better error handling
            try:
                metrics['sensitivity'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            except Exception as e:
                print(f"Error calculating sensitivity for Chronic Condition Risk: {e}")
                metrics['sensitivity'] = 0
        
        elif task == 'Supplement Recommendation':
            # Top-3 Accuracy + MRR (Mean Reciprocal Rank)
            try:
                # More robust implementation for Top-3 Accuracy
                top3_correct = 0
                for i, probs in enumerate(y_probs):
                    top3_indices = np.argsort(probs)[-3:]
                    if y_true[i] in top3_indices:
                        top3_correct += 1
                metrics['top3_accuracy'] = top3_correct / len(y_true)
            except Exception as e:
                print(f"Error calculating top-3 accuracy for Supplement Recommendation: {e}")
                metrics['top3_accuracy'] = 0
            
            try:
                # Improved MRR calculation with error handling
                mrr_sum = 0
                for i, probs in enumerate(y_probs):
                    ranks = np.argsort(probs)[::-1]
                    rank_indices = np.where(ranks == y_true[i])[0]
                    if len(rank_indices) > 0:
                        rank = rank_indices[0] + 1
                        mrr_sum += 1 / rank
                metrics['mrr'] = mrr_sum / len(y_true)
            except Exception as e:
                print(f"Error calculating MRR for Supplement Recommendation: {e}")
                metrics['mrr'] = 0
        
        # Common metrics for all tasks
        metrics['accuracy'] = task_accuracies[task]
        
        task_metrics[task] = metrics
    
    return avg_loss, task_avg_losses, task_accuracies, task_metrics

def plot_confusion_matrices(model, test_loader, label_encoders, device, save_dir):
    """Plot confusion matrices for each task."""
    model.eval()
    
    # For storing predictions and targets
    all_predictions = {task: [] for task in label_encoders.keys()}
    all_targets = {task: [] for task in label_encoders.keys()}
    
    with torch.no_grad():
        for features, targets in tqdm(test_loader, desc="Generating confusion matrices"):
            # Move data to device
            features = features.to(device)
            
            # Forward pass
            outputs = model(features)
            
            # Get predictions
            for task, output in outputs.items():
                _, predicted = torch.max(output, 1)
                all_predictions[task].extend(predicted.cpu().numpy())
                all_targets[task].extend(targets[task].cpu().numpy())
    
    # Plot confusion matrices
    for task, encoder in label_encoders.items():
        try:
            y_true = np.array(all_targets[task])
            y_pred = np.array(all_predictions[task])
            
            # Get class names
            idx_to_class = {idx: cls for cls, idx in encoder.items()}
            class_names = [idx_to_class[i] for i in range(len(encoder))]
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Plot
            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - {task}')
            plt.colorbar()
            tick_marks = np.arange(len(class_names))
            
            # Use shorter class names if they're too long
            display_names = [name[:20] + '...' if len(name) > 20 else name for name in class_names]
            
            plt.xticks(tick_marks, display_names, rotation=45, ha='right')
            plt.yticks(tick_marks, display_names)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            
            # Save figure with Windows-friendly path
            plt.savefig(save_dir / f'confusion_matrix_{task.replace(" ", "_")}.png')
            plt.close()
        except Exception as e:
            print(f"Error creating confusion matrix for {task}: {e}")

def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, save_dir):
    """Plot learning curves for loss and accuracy."""
    # Plot loss curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    for task in train_accs[0].keys():
        task_train_accs = [accs[task] for accs in train_accs]
        task_val_accs = [accs[task] for accs in val_accs]
        plt.plot(task_train_accs, label=f'{task} Train')
        plt.plot(task_val_accs, label=f'{task} Val')
    
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'learning_curves.png')
    plt.close()

def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-4, weight_decay=1e-5, patience=10):
    """Train the multi-task model."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Define loss functions for each task
    criterion = {task: nn.CrossEntropyLoss() for task in model.task_heads.keys()}
    
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Define early stopping with proper Windows path
    checkpoint_path = MODELS_DIR / 'best_model.pt'
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint_path)
    
    # For storing metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Train
        train_loss, train_task_losses, train_task_accs = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_task_losses, val_task_accs, val_task_metrics = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_task_accs)
        val_accs.append(val_task_accs)
        
        # Print metrics
        epoch_time = time.time() - epoch_start
        print(f'Epoch time: {epoch_time:.2f}s, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        for task in train_task_accs.keys():
            print(f'{task} - Train Acc: {train_task_accs[task]:.4f}, Val Acc: {val_task_accs[task]:.4f}')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, MODELS_DIR / f'checkpoint_epoch_{epoch+1}.pt')
            print(f"Checkpoint saved at epoch {epoch+1}")
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print('Early stopping triggered')
            break
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
    
    # Load best model
    try:
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded best model from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading best model: {e}")
    
    return model, train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, test_loader, label_encoders):
    """Evaluate the model on the test set."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Define loss functions for each task
    criterion = {task: nn.CrossEntropyLoss() for task in model.task_heads.keys()}
    
    # Evaluate
    test_loss, test_task_losses, test_task_accs, test_task_metrics = validate(model, test_loader, criterion, device)
    
    # Print metrics
    print(f'Test Loss: {test_loss:.4f}')
    for task, metrics in test_task_metrics.items():
        print(f'{task} Metrics:')
        for metric_name, metric_value in metrics.items():
            # Handle different types of metric values
            if isinstance(metric_value, np.ndarray):
                print(f'  {metric_name}: {metric_value.tolist()}')
            else:
                print(f'  {metric_name}: {metric_value}')
    
    # Plot confusion matrices
    plot_confusion_matrices(model, test_loader, label_encoders, device, RESULTS_DIR)
    
    return test_task_metrics

def main():
    """Main function to train and evaluate the model."""
    try:
        print("Starting multi-task learning model training...")
        print(f"PyTorch version: {torch.__version__}")
        
        # Check for GPU availability
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        else:
            print("No GPU available, using CPU")
        
        # Verify processed data exists
        train_file = PROCESSED_DIR / 'train.csv'
        val_file = PROCESSED_DIR / 'val.csv'
        test_file = PROCESSED_DIR / 'test.csv'
        
        if not (train_file.exists() and val_file.exists() and test_file.exists()):
            print(f"WARNING: Processed data files not found in {PROCESSED_DIR}")
            print("Please run the data_preprocessing.py script first")
            return
        
        # Create dataloaders
        print("Creating dataloaders...")
        train_loader, val_loader, test_loader, feature_dim, label_encoders = create_dataloaders(BATCH_SIZE)
        
        # Get number of classes for each task
        num_classes = {task: len(encoder) for task, encoder in label_encoders.items()}
        print(f"Number of classes for each task: {num_classes}")
        
        # Create model
        print("Creating model...")
        model = create_model(feature_dim, num_classes)
        
        # Train model
        print(f"Training model for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE}...")
        model, train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, val_loader, 
            num_epochs=NUM_EPOCHS, 
            lr=LEARNING_RATE, 
            weight_decay=WEIGHT_DECAY, 
            patience=EARLY_STOPPING_PATIENCE
        )
        
        # Plot learning curves
        print("Plotting learning curves...")
        plot_learning_curves(train_losses, val_losses, train_accs, val_accs, RESULTS_DIR)
        
        # Evaluate model
        print("Evaluating model on test set...")
        test_metrics = evaluate_model(model, test_loader, label_encoders)
        
        # Save metrics
        print("Saving test metrics...")
        with open(RESULTS_DIR / 'test_metrics.pkl', 'wb') as f:
            pickle.dump(test_metrics, f)
        
        print(f"Training completed. Results saved to {RESULTS_DIR}")
        
    except Exception as e:
        import traceback
        print(f"Error in training process: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()