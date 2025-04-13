#!/usr/bin/env python3
"""
Test script to verify that the trained model can be loaded correctly.
"""

import torch
import pickle
import pandas as pd
from pathlib import Path
import os
import sys

# Add current directory to path to import model
sys.path.append(os.getcwd())

# Import the model class
try:
    from model import MultiTaskModel
    print("✅ Successfully imported MultiTaskModel class")
except ImportError as e:
    print(f"❌ Error importing model: {e}")
    sys.exit(1)

# Set paths
CURRENT_DIR = Path(os.getcwd())
DATA_DIR = CURRENT_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = CURRENT_DIR / 'models'

# Constants for model
EMBEDDING_DIM = 32
NUM_HEADS = 4
NUM_LAYERS = 3
DROPOUT = 0.1

def test_model_loading():
    """Test loading the model and its components."""
    print("\n=== Testing Model Loading ===\n")
    
    # Check if model file exists
    model_path = MODELS_DIR / 'best_model.pt'
    if not model_path.exists():
        print(f"❌ Model file not found at {model_path}")
        return False
    print(f"✅ Found model file at {model_path}")
    
    # Check if label encoders file exists
    encoders_path = MODELS_DIR / 'label_encoders.pkl'
    if not encoders_path.exists():
        print(f"❌ Label encoders file not found at {encoders_path}")
        return False
    print(f"✅ Found label encoders file at {encoders_path}")
    
    # Check if target columns file exists
    target_columns_path = PROCESSED_DIR / 'target_columns.pkl'
    if not target_columns_path.exists():
        print(f"❌ Target columns file not found at {target_columns_path}")
        return False
    print(f"✅ Found target columns file at {target_columns_path}")
    
    # Check if training data exists
    train_data_path = PROCESSED_DIR / 'train.csv'
    if not train_data_path.exists():
        print(f"❌ Training data file not found at {train_data_path}")
        return False
    print(f"✅ Found training data at {train_data_path}")
    
    # Try loading label encoders
    try:
        with open(encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
        print(f"✅ Successfully loaded label encoders")
        print(f"   Tasks: {list(label_encoders.keys())}")
    except Exception as e:
        print(f"❌ Error loading label encoders: {e}")
        return False
    
    # Try loading target columns
    try:
        with open(target_columns_path, 'rb') as f:
            target_columns = pickle.load(f)
        print(f"✅ Successfully loaded target columns")
        print(f"   Target columns: {target_columns}")
    except Exception as e:
        print(f"❌ Error loading target columns: {e}")
        return False
    
    # Try loading training data
    try:
        train_data = pd.read_csv(train_data_path)
        feature_columns = [col for col in train_data.columns if col not in target_columns.values()]
        print(f"✅ Successfully loaded training data")
        print(f"   Number of features: {len(feature_columns)}")
        print(f"   Number of samples: {len(train_data)}")
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return False
    
    # Try creating and loading model
    try:
        # Get number of classes for each task
        num_classes = {task: len(encoder) for task, encoder in label_encoders.items()}
        print(f"   Number of classes for each task: {num_classes}")
        
        # Create model
        model = MultiTaskModel(
            input_dim=len(feature_columns),
            embedding_dim=EMBEDDING_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            num_classes=num_classes
        )
        print(f"✅ Successfully created model instance")
        
        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"✅ Successfully loaded model weights")
        
        # Try a simple prediction with random data
        print("\n=== Testing Model Prediction ===\n")
        sample_input = torch.rand(1, len(feature_columns))
        with torch.no_grad():
            outputs = model(sample_input)
        
        print(f"✅ Successfully ran prediction")
        for task, output in outputs.items():
            print(f"   {task} output shape: {output.shape}")
        
        print("\n=== All Tests Passed ===\n")
        return True
    except Exception as e:
        print(f"❌ Error creating or loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_loading()