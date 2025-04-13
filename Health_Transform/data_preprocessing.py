#!/usr/bin/env python3
"""
Data preprocessing script for health data classification project.
This script preprocesses the health data for the multi-task learning model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import os

# Set paths for Windows environment
CURRENT_DIR = Path(os.getcwd())
DATA_DIR = CURRENT_DIR / 'data'
RESULTS_DIR = CURRENT_DIR / 'results'
PROCESSED_DIR = DATA_DIR / 'processed'

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# Define the target columns for each task
TARGET_COLUMNS = {
    'Microbiome Status': 'Current status of microbiota',
    'Obesity Risk': 'BMI',  # We'll create categories based on BMI
    'Digestive Health': 'Intestinal health indicators',
    'Chronic Condition Risk': 'Diagnosed conditions',
    'Supplement Recommendation': 'Supplement Plan - Recommended products'
}

def load_data():
    """Load the health data CSV file."""
    file_path = DATA_DIR / 'health_data_10000_chunk.csv'
    print(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"ERROR: The file {file_path} does not exist. Please place your CSV file in the {DATA_DIR} directory.")
        return None

def preprocess_data(df):
    """Preprocess the data for the multi-task learning model."""
    if df is None:
        print("No data to preprocess. Please check the data file location.")
        return None
    
    print("\n=== Preprocessing Data ===")
    
    # Make a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # 1. Create target variables for each task
    print("\n1. Creating target variables for each task...")
    
    # Task 1: Microbiome Status (already categorical)
    if TARGET_COLUMNS['Microbiome Status'] in processed_df.columns:
        print(f"Task 1 - Microbiome Status: {processed_df[TARGET_COLUMNS['Microbiome Status']].value_counts().to_dict()}")
    else:
        print(f"Warning: Column '{TARGET_COLUMNS['Microbiome Status']}' not found in dataset")
    
    # Task 2: Obesity Risk (create categories based on BMI)
    if 'BMI' in processed_df.columns:
        def categorize_bmi(bmi):
            if pd.isna(bmi):
                return 'Unknown'
            elif bmi < 18.5:
                return 'Underweight'
            elif bmi < 25:
                return 'Normal'
            elif bmi < 30:
                return 'Overweight'
            else:
                return 'Obese'
        
        processed_df['Obesity Risk'] = processed_df['BMI'].apply(categorize_bmi)
        print(f"Task 2 - Obesity Risk: {processed_df['Obesity Risk'].value_counts().to_dict()}")
    else:
        print(f"Warning: Column 'BMI' not found in dataset")
        processed_df['Obesity Risk'] = 'Unknown'
    
    # Task 3: Digestive Health (already categorical)
    if TARGET_COLUMNS['Digestive Health'] in processed_df.columns:
        print(f"Task 3 - Digestive Health: {processed_df[TARGET_COLUMNS['Digestive Health']].value_counts().to_dict()}")
    else:
        print(f"Warning: Column '{TARGET_COLUMNS['Digestive Health']}' not found in dataset")
    
    # Task 4: Chronic Condition Risk (extract first condition for simplicity)
    if TARGET_COLUMNS['Chronic Condition Risk'] in processed_df.columns:
        # Handle NaN values before string operations
        processed_df['Chronic Condition Risk'] = processed_df[TARGET_COLUMNS['Chronic Condition Risk']].fillna('Unknown')
        # Handle non-string columns
        if processed_df[TARGET_COLUMNS['Chronic Condition Risk']].dtype != 'object':
            processed_df['Chronic Condition Risk'] = processed_df[TARGET_COLUMNS['Chronic Condition Risk']].astype(str)
        processed_df['Chronic Condition Risk'] = processed_df[TARGET_COLUMNS['Chronic Condition Risk']].str.split(',').str[0]
        print(f"Task 4 - Chronic Condition Risk: {processed_df['Chronic Condition Risk'].value_counts().head(5).to_dict()}")
    else:
        print(f"Warning: Column '{TARGET_COLUMNS['Chronic Condition Risk']}' not found in dataset")
        processed_df['Chronic Condition Risk'] = 'Unknown'
    
    # Task 5: Supplement Recommendation (extract first supplement for simplicity)
    if TARGET_COLUMNS['Supplement Recommendation'] in processed_df.columns:
        # Handle NaN values before string operations
        processed_df['Supplement Recommendation'] = processed_df[TARGET_COLUMNS['Supplement Recommendation']].fillna('Unknown')
        # Handle non-string columns
        if processed_df[TARGET_COLUMNS['Supplement Recommendation']].dtype != 'object':
            processed_df['Supplement Recommendation'] = processed_df[TARGET_COLUMNS['Supplement Recommendation']].astype(str)
        processed_df['Supplement Recommendation'] = processed_df[TARGET_COLUMNS['Supplement Recommendation']].str.split(',').str[0]
        print(f"Task 5 - Supplement Recommendation: {processed_df['Supplement Recommendation'].value_counts().head(5).to_dict()}")
    else:
        print(f"Warning: Column '{TARGET_COLUMNS['Supplement Recommendation']}' not found in dataset")
        processed_df['Supplement Recommendation'] = 'Unknown'
    
    # 2. Handle categorical features
    print("\n2. Encoding categorical features...")
    
    # Identify categorical columns (excluding target columns and address)
    categorical_cols = processed_df.select_dtypes(include=['object']).columns
    
    # Exclude the target columns we just created and any residential address column
    exclude_cols = ['Residential Address'] + list(TARGET_COLUMNS.values()) + \
                  ['Obesity Risk', 'Chronic Condition Risk', 'Supplement Recommendation']
    
    # Filter to only include columns that exist in the dataframe
    exclude_cols = [col for col in exclude_cols if col in processed_df.columns]
    
    # Final list of categorical columns to encode
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    print(f"Categorical columns to encode: {categorical_cols[:5]}... (total: {len(categorical_cols)})")
    
    # One-hot encode categorical features
    for col in categorical_cols:
        # Skip columns with too many missing values
        if processed_df[col].isna().sum() / len(processed_df) > 0.5:
            print(f"  Skipping column {col} due to >50% missing values")
            continue
            
        # For columns with many unique values, we'll just keep the top 10 categories
        if processed_df[col].nunique() > 10:
            top_categories = processed_df[col].value_counts().nlargest(10).index
            processed_df[col] = processed_df[col].apply(lambda x: x if x in top_categories else 'Other')
        
        try:
            # Create dummy variables
            dummies = pd.get_dummies(processed_df[col], prefix=col, drop_first=False)
            processed_df = pd.concat([processed_df, dummies], axis=1)
        except Exception as e:
            print(f"  Error encoding {col}: {str(e)}")
    
    # 3. Normalize numeric features
    print("\n3. Normalizing numeric features...")
    numeric_cols = processed_df.select_dtypes(include=['int64', 'float64']).columns
    
    # Skip columns we don't want to normalize
    exclude_from_norm = ['Obesity Risk']
    numeric_cols = [col for col in numeric_cols if col not in exclude_from_norm]
    
    print(f"Numeric columns to normalize: {numeric_cols[:5]}... (total: {len(numeric_cols)})")
    
    # Apply min-max normalization
    for col in numeric_cols:
        # Skip columns with too many missing values
        if processed_df[col].isna().sum() / len(processed_df) > 0.5:
            print(f"  Skipping column {col} due to >50% missing values")
            continue
            
        try:
            min_val = processed_df[col].min()
            max_val = processed_df[col].max()
            
            # Avoid division by zero
            if min_val == max_val:
                processed_df[f'{col}_normalized'] = 0  # Set to constant if min==max
            else:
                processed_df[f'{col}_normalized'] = (processed_df[col] - min_val) / (max_val - min_val)
        except Exception as e:
            print(f"  Error normalizing {col}: {str(e)}")
    
    # 4. Handle boolean features
    print("\n4. Converting boolean features...")
    bool_cols = processed_df.select_dtypes(include=['bool']).columns
    print(f"Boolean columns to convert: {list(bool_cols)[:5]}... (total: {len(bool_cols)})")
    
    for col in bool_cols:
        processed_df[col] = processed_df[col].astype(int)
    
    # 5. Drop unnecessary columns
    print("\n5. Dropping unnecessary columns...")
    # We'll drop the original categorical columns since we've one-hot encoded them
    address_cols = [col for col in processed_df.columns if 'address' in col.lower()]
    cols_to_drop = address_cols + categorical_cols
    
    # Make sure all columns exist before dropping
    cols_to_drop = [col for col in cols_to_drop if col in processed_df.columns]
    
    processed_df = processed_df.drop(columns=cols_to_drop)
    
    # 6. Handle missing values in the final dataset
    print("\n6. Handling remaining missing values...")
    # Fill missing values in numeric columns with the mean
    numeric_cols = processed_df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if processed_df[col].isna().sum() > 0:
            processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
    
    # Fill missing values in categorical columns with the most frequent value
    cat_cols = processed_df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if processed_df[col].isna().sum() > 0:
            processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
    
    print(f"Final dataframe shape: {processed_df.shape}")
    
    return processed_df

def split_data(df, test_size=0.2, val_size=0.25):
    """Split the data into train, validation, and test sets."""
    if df is None:
        print("No data to split. Please check the data file location.")
        return None, None, None
        
    print("\n=== Splitting Data ===")
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into train and temp (val + test)
    train_size = 1 - test_size
    train_df = df.iloc[:int(train_size * len(df))]
    temp_df = df.iloc[int(train_size * len(df)):]
    
    # Split temp into val and test
    val_df = temp_df.iloc[:int(val_size * len(temp_df))]
    test_df = temp_df.iloc[int(val_size * len(temp_df)):]
    
    print(f"Train set: {train_df.shape}")
    print(f"Validation set: {val_df.shape}")
    print(f"Test set: {test_df.shape}")
    
    return train_df, val_df, test_df

def save_processed_data(train_df, val_df, test_df, target_columns):
    """Save the processed data to disk."""
    if train_df is None or val_df is None or test_df is None:
        print("Missing data sets. Cannot save processed data.")
        return
        
    print("\n=== Saving Processed Data ===")
    
    # Save the dataframes
    train_df.to_csv(PROCESSED_DIR / 'train.csv', index=False)
    val_df.to_csv(PROCESSED_DIR / 'val.csv', index=False)
    test_df.to_csv(PROCESSED_DIR / 'test.csv', index=False)
    
    # Save the target columns
    with open(PROCESSED_DIR / 'target_columns.pkl', 'wb') as f:
        pickle.dump(target_columns, f)
    
    print(f"Processed data saved to {PROCESSED_DIR}")

def main():
    """Main function to run the data preprocessing."""
    # Print directory information
    print(f"Current working directory: {CURRENT_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Processed directory: {PROCESSED_DIR}")
    
    # Load the data
    df = load_data()
    
    if df is not None:
        # Preprocess the data
        processed_df = preprocess_data(df)
        
        # Split the data
        train_df, val_df, test_df = split_data(processed_df)
        
        # Define the target columns for the model
        target_columns = {
            'Microbiome Status': 'Current status of microbiota',
            'Obesity Risk': 'Obesity Risk',
            'Digestive Health': 'Intestinal health indicators',
            'Chronic Condition Risk': 'Chronic Condition Risk',
            'Supplement Recommendation': 'Supplement Recommendation'
        }
        
        # Save the processed data
        save_processed_data(train_df, val_df, test_df, target_columns)
        
        print("\nData preprocessing completed. Processed data saved to the data/processed directory.")
    else:
        print("\nData preprocessing could not be completed due to missing data file.")
        print("Please ensure your health_data_10000_chunk.csv file is in the correct location.")

if __name__ == "__main__":
    main()