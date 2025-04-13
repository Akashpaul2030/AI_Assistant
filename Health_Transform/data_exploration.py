#!/usr/bin/env python3
"""
Data exploration script for health data classification project.
This script loads the health data CSV file and performs initial exploration.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Set paths - Using relative paths that work on Windows
# Current directory is assumed to be 'G:\savory_haven_chatbot\Health_Transform'
CURRENT_DIR = Path(os.getcwd())
DATA_DIR = CURRENT_DIR / 'data'
RESULTS_DIR = CURRENT_DIR / 'results'

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Load the dataset
def load_data():
    """Load the health data CSV file."""
    file_path = DATA_DIR / 'health_data_10000_chunk.csv'
    print(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"ERROR: The file {file_path} does not exist. Please place your CSV file in the {DATA_DIR} directory.")
        print(f"Creating the data directory at: {DATA_DIR}")
        return None

# Explore the dataset
def explore_data(df):
    """Perform initial exploration of the dataset."""
    if df is None:
        print("No data to explore. Please check the data file location.")
        return None, None, None
        
    # Basic information
    print("\n=== Dataset Information ===")
    print(f"Shape: {df.shape}")
    print(f"Number of samples: {df.shape[0]}")
    print(f"Number of features: {df.shape[1]}")
    
    # Column names and types
    print("\n=== Column Names and Types ===")
    print(df.dtypes)
    
    # Check for missing values
    print("\n=== Missing Values ===")
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percent
    })
    print(missing_df[missing_df['Missing Values'] > 0])
    
    # Summary statistics for numerical columns
    print("\n=== Summary Statistics for Numerical Columns ===")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print(df[numeric_cols].describe())
    
    # Value counts for categorical columns (first few)
    print("\n=== Value Counts for Selected Categorical Columns ===")
    categorical_cols = df.select_dtypes(include=['object']).columns[:5]  # First 5 categorical columns
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts().head())
    
    return missing_df, numeric_cols, categorical_cols

# Identify potential target columns for the 5 classification tasks
def identify_target_columns(df):
    """Identify potential target columns for the 5 classification tasks."""
    if df is None:
        print("No data to analyze. Please check the data file location.")
        return None
        
    print("\n=== Potential Target Columns ===")
    
    # Based on the requirements, we need to identify columns for:
    # 1. Microbiome Status (3 classes)
    # 2. Obesity Risk (4 classes)
    # 3. Digestive Health (3 classes)
    # 4. Chronic Condition Risk (4 classes)
    # 5. Supplement Recommendation (5 classes)
    
    # Let's look for columns that might match these tasks
    potential_targets = {
        'Microbiome Status': ['Current status of microbiota'],
        'Obesity Risk': ['BMI', 'Weight (kg)'],
        'Digestive Health': ['Intestinal health indicators', 'Presence of bloating', 
                            'Presence of gas', 'Presence of abdominal pain', 
                            'Difficult digestion', 'Frequency of bowel movements', 
                            'Stool consistency (Bristol scale)'],
        'Chronic Condition Risk': ['Medical conditions', 'Diagnosed conditions', 
                                  'Family history of diseases'],
        'Supplement Recommendation': ['Supplement Plan - Recommended products']
    }
    
    # Check if these columns exist in the dataset
    for task, columns in potential_targets.items():
        print(f"\n{task}:")
        for col in columns:
            if col in df.columns:
                print(f"  - {col} (exists)")
                # Show value counts for the first few values
                print(f"    Values: {df[col].value_counts().head(3).to_dict()}")
            else:
                print(f"  - {col} (does not exist)")
    
    return potential_targets

# Visualize distributions of key features
def visualize_key_features(df, numeric_cols, save_dir=None):
    """Visualize distributions of key numerical features."""
    if df is None or numeric_cols is None:
        print("No data or columns to visualize. Please check the data file location.")
        return
        
    if save_dir is None:
        save_dir = RESULTS_DIR
    
    # Create a directory for visualizations if it doesn't exist
    viz_dir = save_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    # Select a subset of numeric columns to visualize
    if len(numeric_cols) > 0:
        cols_to_viz = numeric_cols[:min(5, len(numeric_cols))]  # First 5 numeric columns or fewer
        
        # Histograms for numeric features
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(cols_to_viz, 1):
            plt.subplot(2, 3, i)
            plt.hist(df[col].dropna(), bins=20, alpha=0.7)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'numeric_distributions.png')
        print(f"Saved numeric distributions plot to {viz_dir / 'numeric_distributions.png'}")
    else:
        print("No numeric columns found for visualization.")
    
    # BMI distribution (if available)
    if 'BMI' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df['BMI'].dropna(), bins=30, alpha=0.7)
        plt.title('Distribution of BMI')
        plt.xlabel('BMI')
        plt.ylabel('Frequency')
        plt.axvline(18.5, color='r', linestyle='--', label='Underweight (<18.5)')
        plt.axvline(25, color='g', linestyle='--', label='Normal (18.5-25)')
        plt.axvline(30, color='y', linestyle='--', label='Overweight (25-30)')
        plt.axvline(35, color='orange', linestyle='--', label='Obese Class I (30-35)')
        plt.axvline(40, color='purple', linestyle='--', label='Obese Class II (35-40)')
        plt.legend()
        plt.savefig(viz_dir / 'bmi_distribution.png')
        print(f"Saved BMI distribution plot to {viz_dir / 'bmi_distribution.png'}")

def main():
    """Main function to run the data exploration."""
    # Check if directories exist
    print(f"Current working directory: {CURRENT_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    
    # Load the data
    df = load_data()
    
    if df is not None:
        # Explore the data
        missing_df, numeric_cols, categorical_cols = explore_data(df)
        
        # Identify potential target columns
        potential_targets = identify_target_columns(df)
        
        # Visualize key features
        visualize_key_features(df, numeric_cols)
        
        print("\nData exploration completed. Results saved to the results directory.")
    else:
        print("\nData exploration could not be completed due to missing data file.")
        print("Please ensure your health_data_10000_chunk.csv file is in the correct location.")

if __name__ == "__main__":
    main()