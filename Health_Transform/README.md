#!/usr/bin/env python3
"""
README for Health Data Multi-Task Learning Project

This project implements a transformer-based multi-task learning pipeline for health data classification.
The pipeline handles 5 classification tasks:
1. Microbiome Status (3 classes)
2. Obesity Risk (4 classes)
3. Digestive Health (3 classes)
4. Chronic Condition Risk (4 classes)
5. Supplement Recommendation (5 classes)
"""

# Project Structure
# ----------------
# health_ml_project/
# ├── data/
# │   ├── health_data_10000_chunk.csv      # Original dataset
# │   └── processed/                       # Processed data
# │       ├── train.csv                    # Training set
# │       ├── val.csv                      # Validation set
# │       ├── test.csv                     # Test set
# │       └── target_columns.pkl           # Target column mappings
# ├── models/                              # Saved models
# │   ├── model_architecture.pt            # Model architecture
# │   ├── best_model.pt                    # Best model weights
# │   └── label_encoders.pkl               # Label encoders
# ├── results/                             # Results and visualizations
# │   ├── visualizations/                  # Data exploration visualizations
# │   ├── confusion_matrix_*.png           # Confusion matrices
# │   ├── learning_curves.png              # Learning curves
# │   ├── sample_predictions.png           # Sample predictions
# │   └── test_metrics.pkl                 # Test metrics
# └── src/                                 # Source code
#     ├── data_exploration.py              # Data exploration script
#     ├── data_preprocessing.py            # Data preprocessing script
#     ├── model.py                         # Model architecture
#     ├── train.py                         # Training script
#     └── predict.py                       # Prediction script

# Installation
# -----------
# This project requires Python 3.8+ and the following packages:
# - PyTorch
# - pandas
# - numpy
# - scikit-learn
# - matplotlib
# - seaborn
# - tqdm
#
# Install dependencies:
# ```
# pip install torch torchvision torchaudio pandas scikit-learn matplotlib seaborn numpy tqdm
# ```

# Usage
# -----
# 1. Data Exploration:
#    ```
#    python src/data_exploration.py
#    ```
#
# 2. Data Preprocessing:
#    ```
#    python src/data_preprocessing.py
#    ```
#
# 3. Model Training:
#    ```
#    python src/train.py
#    ```
#
# 4. Making Predictions:
#    ```
#    python src/predict.py
#    ```

# Model Architecture
# -----------------
# The model uses a transformer-based architecture for tabular data:
# - TabTransformerEncoder: A transformer encoder adapted for tabular data
# - MultiTaskModel: A model with a shared transformer encoder and task-specific heads
#
# The architecture includes:
# - Feature embedding layer
# - Positional encoding
# - Transformer encoder layers
# - Task-specific classification heads

# Training Pipeline
# ---------------
# The training pipeline includes:
# - Multi-task cross-entropy loss
# - AdamW optimizer
# - ReduceLROnPlateau learning rate scheduler
# - Early stopping
# - Comprehensive evaluation metrics for each task

# Evaluation Metrics
# ----------------
# Task-specific metrics:
# - Microbiome Status: Balanced Accuracy, F1, Cohen's Kappa
# - Obesity Risk: Weighted F1, Precision/Recall per class
# - Digestive Health: Macro F1, AUC-ROC
# - Chronic Condition Risk: AUC-PR, Sensitivity
# - Supplement Recommendation: Top-3 Accuracy, MRR

# Results Visualization
# -------------------
# The pipeline generates:
# - Confusion matrices for each task
# - Learning curves (loss and accuracy)
# - Sample predictions visualization
# - Comprehensive metrics report

# Next Steps
# ---------
# 1. Hyperparameter Tuning:
#    - Adjust embedding dimensions, number of heads, and layers
#    - Experiment with different learning rates and batch sizes
#
# 2. Model Interpretability:
#    - Implement attention visualization
#    - Add feature importance analysis
#
# 3. Deployment:
#    - Convert model to ONNX format for deployment
#    - Create a simple API for making predictions
