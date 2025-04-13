#!/usr/bin/env python3
"""
Transformer model implementation for tabular data classification.
This script defines the transformer-based architecture for multi-task learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import os

# Set paths
# Set paths for Windows environment
CURRENT_DIR = Path(os.getcwd())
DATA_DIR = CURRENT_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = CURRENT_DIR / 'models'

# Define constants
EMBEDDING_DIM = 32
NUM_HEADS = 4
NUM_LAYERS = 3
DROPOUT = 0.1
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5

# Define the number of classes for each task
NUM_CLASSES = {
    'Microbiome Status': 2,  # Optimal, Suboptimal
    'Obesity Risk': 4,       # Underweight, Normal, Overweight, Obese
    'Digestive Health': 2,   # Optimal, Suboptimal
    'Chronic Condition Risk': 5,  # Top 5 conditions
    'Supplement Recommendation': 5  # Top 5 supplements
}

class HealthDataset(Dataset):
    """Dataset class for health data."""
    
    def __init__(self, csv_file, target_columns):
        """
        Args:
            csv_file (str): Path to the CSV file with the processed data.
            target_columns (dict): Dictionary mapping task names to target column names.
        """
        self.data = pd.read_csv(csv_file)
        self.target_columns = target_columns
        
        # Separate features and targets
        self.features = self.data.drop(columns=list(target_columns.values()))
        
        # Create label encoders for categorical targets
        self.label_encoders = {}
        self.targets = {}
        
        for task, col in target_columns.items():
            # Get unique classes
            unique_classes = self.data[col].unique()
            # Create mapping from class to index
            class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
            self.label_encoders[task] = class_to_idx
            # Encode targets
            self.targets[task] = self.data[col].map(class_to_idx).values
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get features and ensure they're numeric before conversion to torch tensor
        feature_values = self.features.iloc[idx].values
        
        # Handle non-numeric values by converting numpy array properly
        try:
            # First attempt - direct conversion
            features = torch.tensor(feature_values, dtype=torch.float32)
        except TypeError:
            # If that fails, try converting with numpy first
            # This handles numpy.object_ arrays that might contain mixed types
            try:
                # Convert to stacked float numpy array
                feature_values = np.vstack(feature_values).astype(np.float32)
                features = torch.from_numpy(feature_values).float()
            except (ValueError, TypeError):
                # If specific columns are causing issues, we need more detailed handling
                # Convert each value individually with error handling
                numeric_values = []
                for val in feature_values:
                    try:
                        # Try to convert to float
                        numeric_values.append(float(val))
                    except (ValueError, TypeError):
                        # Replace non-numeric with 0.0
                        numeric_values.append(0.0)
                features = torch.tensor(numeric_values, dtype=torch.float32)
        
        # Get targets for each task
        targets = {task: torch.tensor(self.targets[task][idx], dtype=torch.long) 
                  for task in self.target_columns.keys()}
        
        return features, targets
    
    def get_feature_dim(self):
        """Get the dimension of the feature vector."""
        return self.features.shape[1]
    
    def get_label_encoders(self):
        """Get the label encoders for each task."""
        return self.label_encoders

class TabTransformerEncoder(nn.Module):
    """Transformer encoder for tabular data."""
    
    def __init__(self, input_dim, embedding_dim, num_heads, num_layers, dropout):
        """
        Args:
            input_dim (int): Dimension of the input features.
            embedding_dim (int): Dimension of the embeddings.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            dropout (float): Dropout rate.
        """
        super(TabTransformerEncoder, self).__init__()
        
        # Feature embedding layer - maps each 1D feature to embedding_dim dimensions
        self.feature_embedding = nn.Linear(1, embedding_dim)
        
        # Positional encoding
        self.register_buffer('positional_encoding', self._create_positional_encoding(input_dim, embedding_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output pooling - takes the flattened output and reduces to embedding_dim
        self.pooling = nn.Sequential(
            nn.Linear(embedding_dim * input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU()
        )
    
    def _create_positional_encoding(self, seq_len, d_model):
        """Create positional encoding for the transformer."""
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pos_encoding = torch.zeros(1, seq_len, d_model)
        pos_encoding[0, :, 0::2] = torch.sin(position * div_term)
        pos_encoding[0, :, 1::2] = torch.cos(position * div_term)
        return pos_encoding
    
    def forward(self, x):
        """
        Forward pass through the transformer encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, embedding_dim).
        """
        # Get dimensions
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # For transformer, we need 3D input: (batch_size, seq_len, feature_dim)
        # But our input is (batch_size, seq_len) where each feature is a scalar
        # So we need to unsqueeze to make it (batch_size, seq_len, 1)
        x = x.unsqueeze(-1)  # Now shape is (batch_size, seq_len, 1)
        
        # Embed each feature individually
        embedded_features = []
        for i in range(seq_len):
            # Extract the i-th feature for all samples in the batch
            feature = x[:, i, :]  # Shape: (batch_size, 1)
            # Embed this feature
            embedded = self.feature_embedding(feature)  # Shape: (batch_size, embedding_dim)
            embedded_features.append(embedded)
        
        # Stack all embedded features
        x = torch.stack(embedded_features, dim=1)  # Shape: (batch_size, seq_len, embedding_dim)
        
        # Add positional encoding
        if x.size(1) <= self.positional_encoding.size(1):
            x = x + self.positional_encoding[:, :x.size(1), :]
        else:
            # If input sequence is longer than the pre-computed positional encoding,
            # we need to handle this special case (e.g., by truncating or extending)
            x = x + self.positional_encoding.repeat(1, (x.size(1) // self.positional_encoding.size(1)) + 1, 1)[:, :x.size(1), :]
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # Shape: (batch_size, seq_len, embedding_dim)
        
        # Flatten and pool
        x = x.reshape(batch_size, -1)  # Shape: (batch_size, seq_len * embedding_dim)
        x = self.pooling(x)  # Shape: (batch_size, embedding_dim)
        
        return x

class MultiTaskModel(nn.Module):
    """Multi-task learning model with transformer encoder and task-specific heads."""
    
    def __init__(self, input_dim, embedding_dim, num_heads, num_layers, dropout, num_classes):
        """
        Args:
            input_dim (int): Dimension of the input features.
            embedding_dim (int): Dimension of the embeddings.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            dropout (float): Dropout rate.
            num_classes (dict): Dictionary mapping task names to number of classes.
        """
        super(MultiTaskModel, self).__init__()
        
        # Shared transformer encoder
        self.encoder = TabTransformerEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Task-specific classification heads
        self.task_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.LayerNorm(embedding_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim // 2, n_classes)
            ) for task, n_classes in num_classes.items()
        })
    
    def forward(self, x):
        """
        Forward pass through the multi-task model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            
        Returns:
            dict: Dictionary mapping task names to output logits.
        """
        # Get shared representations from encoder
        shared_repr = self.encoder(x)
        
        # Pass through task-specific heads
        outputs = {task: head(shared_repr) for task, head in self.task_heads.items()}
        
        return outputs

def create_dataloaders(batch_size=64):
    """Create DataLoader objects for train, validation, and test sets."""
    # Load target columns
    import pickle
    with open(PROCESSED_DIR / 'target_columns.pkl', 'rb') as f:
        target_columns = pickle.load(f)
    
    # Create datasets
    train_dataset = HealthDataset(PROCESSED_DIR / 'train.csv', target_columns)
    val_dataset = HealthDataset(PROCESSED_DIR / 'val.csv', target_columns)
    test_dataset = HealthDataset(PROCESSED_DIR / 'test.csv', target_columns)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, train_dataset.get_feature_dim(), train_dataset.get_label_encoders()

def create_model(input_dim, num_classes):
    """Create the multi-task model."""
    model = MultiTaskModel(
        input_dim=input_dim,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_classes=num_classes
    )
    return model

def main():
    """Main function to create and save the model architecture."""
    # Create dataloaders
    train_loader, val_loader, test_loader, feature_dim, label_encoders = create_dataloaders(BATCH_SIZE)
    
    # Get number of classes for each task
    num_classes = {task: len(encoder) for task, encoder in label_encoders.items()}
    print(f"Number of classes for each task: {num_classes}")
    
    # Create model
    model = create_model(feature_dim, num_classes)
    
    # Print model summary
    print(model)
    
    # Save model architecture
    torch.save(model.state_dict(), MODELS_DIR / 'model_architecture.pt')
    
    # Save label encoders
    import pickle
    with open(MODELS_DIR / 'label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    print(f"Model architecture saved to {MODELS_DIR / 'model_architecture.pt'}")
    print(f"Label encoders saved to {MODELS_DIR / 'label_encoders.pkl'}")

if __name__ == "__main__":
    main()