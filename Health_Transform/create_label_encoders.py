#!/usr/bin/env python3
"""
Script to generate label_encoders.pkl file from training data.
"""

import pandas as pd
import pickle
from pathlib import Path
import os

# Set paths
CURRENT_DIR = Path(os.getcwd())
DATA_DIR = CURRENT_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = CURRENT_DIR / 'models'

def create_label_encoders():
    """Create and save label encoders from processed training data."""
    print("Creating label encoders file...")
    
    try:
        # Check if target_columns.pkl exists
        target_columns_path = PROCESSED_DIR / 'target_columns.pkl'
        if not target_columns_path.exists():
            print(f"Error: Target columns file not found at {target_columns_path}")
            return False
        
        # Load target columns
        with open(target_columns_path, 'rb') as f:
            target_columns = pickle.load(f)
        
        print(f"Loaded target columns: {target_columns}")
        
        # Load training data
        train_data_path = PROCESSED_DIR / 'train.csv'
        if not train_data_path.exists():
            print(f"Error: Training data file not found at {train_data_path}")
            return False
        
        train_data = pd.read_csv(train_data_path)
        print(f"Loaded training data with {len(train_data)} rows and {len(train_data.columns)} columns")
        
        # Create label encoders for each task
        label_encoders = {}
        
        for task, column in target_columns.items():
            if column not in train_data.columns:
                print(f"Warning: Column {column} for task {task} not found in training data")
                continue
            
            # Get unique classes
            unique_classes = train_data[column].unique()
            # Create mapping from class to index
            class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
            
            label_encoders[task] = class_to_idx
            print(f"Created encoder for {task} with {len(class_to_idx)} classes: {class_to_idx}")
        
        # Create models directory if it doesn't exist
        MODELS_DIR.mkdir(exist_ok=True)
        
        # Save label encoders
        encoders_path = MODELS_DIR / 'label_encoders.pkl'
        with open(encoders_path, 'wb') as f:
            pickle.dump(label_encoders, f)
        
        print(f"Successfully saved label encoders to {encoders_path}")
        return True
    
    except Exception as e:
        print(f"Error creating label encoders: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    create_label_encoders()