#!/usr/bin/env python3
"""
Script to generate label_encoders.pkl file using the target_columns.pkl file
and the existing best_model.pt file.
"""

import torch
import pickle
import pandas as pd
from pathlib import Path
import os

# Set paths
CURRENT_DIR = Path(os.getcwd())
DATA_DIR = CURRENT_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = CURRENT_DIR / 'models'

def generate_label_encoders():
    """Generate label_encoders.pkl from target_columns.pkl and model."""
    try:
        print("Generating label encoders from target columns and training data...")
        
        # Load target columns
        target_columns_path = PROCESSED_DIR / 'target_columns.pkl'
        if not target_columns_path.exists():
            print(f"Error: Target columns file not found at {target_columns_path}")
            return False
        
        with open(target_columns_path, 'rb') as f:
            target_columns = pickle.load(f)
        
        print(f"Loaded target columns: {target_columns}")
        
        # Load training data
        train_data_path = PROCESSED_DIR / 'train.csv'
        if not train_data_path.exists():
            print(f"Error: Training data file not found at {train_data_path}")
            return False
        
        train_data = pd.read_csv(train_data_path)
        print(f"Loaded training data with {len(train_data)} rows")
        
        # Create label encoders
        label_encoders = {}
        
        for task, column in target_columns.items():
            print(f"Processing task: {task}, column: {column}")
            
            # Check if column exists
            if column not in train_data.columns:
                print(f"Warning: Column {column} not found in training data")
                continue
            
            # Get unique values for this column
            unique_values = sorted(train_data[column].unique())
            print(f"Found {len(unique_values)} unique values for {task}")
            
            # Create mapping
            encoder = {value: i for i, value in enumerate(unique_values)}
            label_encoders[task] = encoder
            
            print(f"Created encoder with {len(encoder)} classes: {encoder}")
        
        # Save label encoders
        encoder_path = MODELS_DIR / 'label_encoders.pkl'
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoders, f)
        
        print(f"Successfully saved label encoders to {encoder_path}")
        
        # Verify that we can load it back
        with open(encoder_path, 'rb') as f:
            loaded_encoders = pickle.load(f)
        
        print(f"Verified loaded encoders: {list(loaded_encoders.keys())}")
        
        return True
    
    except Exception as e:
        print(f"Error generating label encoders: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    generate_label_encoders()