#!/usr/bin/env python3
"""
Streamlit application for health predictions using the multi-task learning model.
"""

import streamlit as st
import torch
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json

# Import the model class
from model import MultiTaskModel

# Set paths
CURRENT_DIR = Path(os.getcwd())
DATA_DIR = CURRENT_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = CURRENT_DIR / 'models'
RESULTS_DIR = CURRENT_DIR / 'results'

# Constants for model
EMBEDDING_DIM = 32
NUM_HEADS = 4
NUM_LAYERS = 3
DROPOUT = 0.1

@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and label encoders."""
    try:
        # Load label encoders
        with open(MODELS_DIR / 'label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        # Display label encoders in logs
        print("Loaded label encoders:", json.dumps(label_encoders, indent=2, default=str))
        
        # Invert label encoders for displaying results
        idx_to_label = {task: {idx: label for label, idx in encoder.items()} 
                       for task, encoder in label_encoders.items()}
        
        # Get feature info from the training data
        train_data = pd.read_csv(PROCESSED_DIR / 'train.csv')
        
        # Load target columns
        with open(PROCESSED_DIR / 'target_columns.pkl', 'rb') as f:
            target_columns = pickle.load(f)
        
        # Get feature columns
        feature_columns = [col for col in train_data.columns if col not in target_columns.values()]
        
        # Get number of classes for each task
        num_classes = {task: len(encoder) for task, encoder in label_encoders.items()}
        print("Number of classes for each task:", num_classes)
        
        # Create model
        model = MultiTaskModel(
            input_dim=len(feature_columns),
            embedding_dim=EMBEDDING_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            num_classes=num_classes
        )
        
        # Load model weights
        model.load_state_dict(torch.load(MODELS_DIR / 'best_model.pt', map_location=torch.device('cpu')))
        model.eval()  # Set model to evaluation mode
        
        return model, label_encoders, idx_to_label, feature_columns, target_columns, train_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None

def determine_column_type(column):
    """Determine if a column is numeric or categorical."""
    # Try to convert to numeric
    try:
        pd.to_numeric(column)
        return 'numeric'
    except:
        # Check if column contains unique values that could be categorical
        if column.nunique() < 10:  # Arbitrary threshold
            return 'categorical'
        else:
            # Try to handle mixed types
            return 'text'

def preprocess_input_for_model(input_data, feature_columns, train_data):
    """Preprocess input data to make it compatible with the model."""
    # Create a DataFrame with the right structure
    processed_data = pd.DataFrame(index=[0], columns=feature_columns)
    
    for feature in feature_columns:
        # Get value from input_data
        value = input_data.get(feature)
        
        # Determine feature type from training data
        feature_type = determine_column_type(train_data[feature])
        
        if feature_type == 'numeric':
            # Convert to float for numeric features
            processed_data[feature] = float(value)
        elif feature_type == 'categorical':
            # For categorical, keep as is
            processed_data[feature] = value
        else:
            # For text/mixed, convert to string
            processed_data[feature] = str(value)
    
    return processed_data

def make_prediction(model, input_data, feature_columns, idx_to_label, train_data):
    """Make predictions using the trained model."""
    # Preprocess input data
    processed_input = preprocess_input_for_model(input_data, feature_columns, train_data)
    
    # Convert to tensor and handle non-numeric data
    tensor_data = []
    for feature in feature_columns:
        try:
            # Try to convert directly to float
            value = float(processed_input[feature].iloc[0])
        except:
            # For non-numeric, use one-hot encoding or dummy value
            # For simplicity, we'll use 0.0 for non-numeric values
            # In a production app, you would need more sophisticated handling
            value = 0.0
        tensor_data.append(value)
    
    features = torch.tensor([tensor_data], dtype=torch.float32)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(features)
    
    # Process results
    predictions = {}
    probabilities = {}
    raw_outputs = {}
    
    for task, output in outputs.items():
        # Store raw model outputs for debugging
        raw_outputs[task] = output.cpu().numpy().tolist()
        
        # Get class probabilities
        probs = torch.softmax(output, dim=1).squeeze(0).numpy()
        
        # Get predicted class
        pred_idx = int(torch.argmax(output, dim=1).item())
        
        # Log the prediction index and corresponding class
        print(f"Task: {task}, Predicted index: {pred_idx}")
        
        # Convert to human-readable label
        if pred_idx in idx_to_label[task]:
            pred_label = idx_to_label[task][pred_idx]
            print(f"  Found label: {pred_label}")
        else:
            # Handle the case where the index is not found
            print(f"  WARNING: Index {pred_idx} not found in label map for {task}")
            print(f"  Available indices: {list(idx_to_label[task].keys())}")
            pred_label = f"Unknown (Index {pred_idx})"
        
        predictions[task] = pred_label
        
        # Create probability mapping with proper handling for missing indices
        task_probs = {}
        for i, prob in enumerate(probs):
            if i in idx_to_label[task]:
                task_probs[idx_to_label[task][i]] = float(prob)
            else:
                task_probs[f"Unknown (Index {i})"] = float(prob)
        
        probabilities[task] = task_probs
    
    # Return all information for debugging
    return predictions, probabilities, raw_outputs

def main():
    """Main function for the Streamlit app."""
    st.title("Health Prediction System")
    st.write("This application uses a multi-task learning model to predict various health outcomes.")
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("Debug Mode", value=True)
    
    # Load model, encoders, and feature columns
    model, label_encoders, idx_to_label, feature_columns, target_columns, train_data = load_model_and_encoders()
    
    if model is None:
        st.error("Failed to load the model. Please check the model path and try again.")
        return
    
    # Display label encoders in debug mode
    if debug_mode:
        st.sidebar.subheader("Label Encoders")
        for task, encoder in label_encoders.items():
            st.sidebar.write(f"**{task}**")
            st.sidebar.json(encoder)
    
    # Display available tasks
    st.subheader("Prediction Tasks")
    task_descriptions = {
        'Microbiome Status': "Predicts the health status of your gut microbiome.",
        'Obesity Risk': "Assesses your risk level for obesity based on health indicators.",
        'Digestive Health': "Evaluates the overall health of your digestive system.",
        'Chronic Condition Risk': "Identifies potential risk for chronic health conditions.",
        'Supplement Recommendation': "Suggests dietary supplements that may benefit your health."
    }
    
    for task, description in task_descriptions.items():
        if task in label_encoders:
            st.write(f"**{task}**: {description}")
    
    # Create form for user input
    st.subheader("Enter Your Health Data")
    st.write("Please provide the following information for prediction:")
    
    # Create input form
    with st.form("prediction_form"):
        # Create a dictionary to store input values
        input_values = {}
        
        # Group features into categories for better user experience
        # In a real app, you would want a more sophisticated grouping
        feature_groups = {
            "Health Metrics": [],
            "Medical History": [],
            "Lifestyle Factors": [],
            "Other Indicators": []
        }
        
        # Simple assignment of features to groups
        for feature in feature_columns:
            if "blood" in feature.lower() or "weight" in feature.lower() or "bmi" in feature.lower():
                feature_groups["Health Metrics"].append(feature)
            elif "condition" in feature.lower() or "disease" in feature.lower() or "history" in feature.lower():
                feature_groups["Medical History"].append(feature)
            elif "diet" in feature.lower() or "exercise" in feature.lower() or "sleep" in feature.lower():
                feature_groups["Lifestyle Factors"].append(feature)
            else:
                feature_groups["Other Indicators"].append(feature)
        
        # Create tabs for feature groups
        tabs = st.tabs(list(feature_groups.keys()))
        
        # Display inputs for each group in its tab
        for i, (group_name, features) in enumerate(feature_groups.items()):
            with tabs[i]:
                if not features:
                    st.write("No features in this category.")
                    continue
                
                # Create columns for a more compact layout
                cols = st.columns(2)
                
                for j, feature in enumerate(features):
                    col = cols[j % 2]
                    
                    # Determine feature type
                    feature_type = determine_column_type(train_data[feature])
                    
                    if feature_type == 'numeric':
                        # Handle numeric features with sliders
                        try:
                            min_val = float(train_data[feature].min())
                            max_val = float(train_data[feature].max())
                            mean_val = float(train_data[feature].mean())
                            
                            # Create slider with appropriate range
                            input_values[feature] = col.slider(
                                f"{feature}",
                                min_value=min_val,
                                max_value=max_val,
                                value=mean_val,
                                step=(max_val - min_val) / 100,  # 100 steps across range
                                help=f"Range: {min_val:.2f} to {max_val:.2f}"
                            )
                        except:
                            # Fallback for numeric features that couldn't be processed
                            input_values[feature] = col.number_input(
                                f"{feature}", 
                                value=0.0,
                                help="Enter a numeric value"
                            )
                    elif feature_type == 'categorical':
                        # Handle categorical features with select boxes
                        options = train_data[feature].unique().tolist()
                        input_values[feature] = col.selectbox(
                            f"{feature}",
                            options=options,
                            help=f"Select from {len(options)} options"
                        )
                    else:
                        # Handle text/mixed features with text inputs
                        default_value = train_data[feature].iloc[0]
                        input_values[feature] = col.text_input(
                            f"{feature}",
                            value=str(default_value),
                            help="Enter appropriate value"
                        )
        
        # Submit button
        submit_button = st.form_submit_button(label="Generate Predictions")
    
    # Make predictions when form is submitted
    if submit_button:
        # Make predictions with additional debugging info
        predictions, probabilities, raw_outputs = make_prediction(model, input_values, feature_columns, idx_to_label, train_data)
        
        # Display predictions
        st.subheader("Prediction Results")
        
        for task, prediction in predictions.items():
            st.write(f"#### {task}")
            st.write(f"**Prediction**: {prediction}")
            
            # Display probabilities as a bar chart
            probs_df = pd.DataFrame({
                'Class': list(probabilities[task].keys()),
                'Probability': list(probabilities[task].values())
            })
            st.bar_chart(probs_df.set_index('Class'))
            
            # Show raw model outputs in debug mode
            if debug_mode:
                st.write("Raw model output:")
                st.write(raw_outputs[task])
                
                st.write("Available labels for this task:")
                st.json(idx_to_label[task])
            
            # Add a divider
            st.markdown("---")
        
        # Add recommendations based on predictions
        st.subheader("Recommendations")
        
        # Try to randomize recommendations for testing
        import random
        if 'Microbiome Status' in predictions:
            status = predictions['Microbiome Status']
            if status == 'Optimal':
                st.write("✅ Your microbiome appears to be in good health. Continue with your current diet and lifestyle.")
            else:
                st.write("⚠️ Your microbiome may benefit from dietary changes. Consider increasing fiber intake and probiotic-rich foods.")
        
        if 'Supplement Recommendation' in predictions:
            recommendation = predictions['Supplement Recommendation']
            st.write(f"💊 Based on your profile, we recommend considering: **{recommendation}**")
            
            # In debug mode, show all possible supplements
            if debug_mode:
                st.write("All possible supplement recommendations:")
                supplement_encoders = {v: k for k, v in label_encoders.get('Supplement Recommendation', {}).items()}
                for idx, name in sorted(supplement_encoders.items()):
                    st.write(f"- {name} (Index: {idx})")
            
            st.write("*Note: Always consult with a healthcare professional before starting any supplement regimen.*")

if __name__ == "__main__":
    main()