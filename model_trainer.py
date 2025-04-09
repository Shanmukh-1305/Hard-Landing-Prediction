import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

def train_models():
    """
    Train ML models for hard landing prediction.
    
    Returns:
    tuple: (classification_model, regression_model) - Trained models
    """
    # Check if data exists, otherwise generate it
    if not os.path.exists('data/flight_data.csv'):
        print("Dataset not found. Generating new dataset...")
        from dataset_generator import generate_flight_data
        df = generate_flight_data(5000)
    else:
        df = pd.read_csv('data/flight_data.csv')
    
    # Prepare data for modeling
    # Features we'll use for prediction during the approach phase
    features = [
        'aircraft_type', 'weather', 'runway_length_ft', 'visibility_miles',
        'wind_speed_knots', 'wind_direction_degrees', 'crosswind_component_knots',
        'approach_speed_knots', 'descent_rate_fpm', 'altitude_ft',
        'glideslope_deviation_dots', 'localizer_deviation_dots',
        'pitch_degrees', 'roll_degrees', 'flap_setting_degrees',
        'gear_position', 'distance_from_threshold_feet', 'throttle_percentage'
    ]
    
    # Target variables
    target_binary = 'hard_landing'  # Binary classification
    target_continuous = 'risk_score'  # Regression
    
    # Split features and targets
    X = df[features]
    y_binary = df[target_binary]
    y_continuous = df[target_continuous]
    
    # Create train/test split
    X_train, X_test, y_binary_train, y_binary_test, y_continuous_train, y_continuous_test = train_test_split(
        X, y_binary, y_continuous, test_size=0.2, random_state=42
    )
    
    # Identify categorical and numerical features
    categorical_features = ['aircraft_type', 'weather', 'gear_position']
    numerical_features = [col for col in features if col not in categorical_features]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Binary classification model (Hard landing: Yes/No)
    clf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Regression model (Risk score prediction)
    reg_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])
    
    # Train models
    print("Training classification model...")
    clf_pipeline.fit(X_train, y_binary_train)
    
    print("Training regression model...")
    reg_pipeline.fit(X_train, y_continuous_train)
    
    # Evaluate models
    # Classification metrics
    y_pred_binary = clf_pipeline.predict(X_test)
    accuracy = accuracy_score(y_binary_test, y_pred_binary)
    precision = precision_score(y_binary_test, y_pred_binary)
    recall = recall_score(y_binary_test, y_pred_binary)
    f1 = f1_score(y_binary_test, y_pred_binary)
    
    print(f"Classification Model Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Regression metrics
    y_pred_continuous = reg_pipeline.predict(X_test)
    mse = mean_squared_error(y_continuous_test, y_pred_continuous)
    r2 = r2_score(y_continuous_test, y_pred_continuous)
    
    print(f"Regression Model Metrics:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save the models
    with open('models/hard_landing_model.pkl', 'wb') as f:
        pickle.dump({
            'classification': clf_pipeline,
            'regression': reg_pipeline,
            'metrics': {
                'classification': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                },
                'regression': {
                    'mse': mse,
                    'r2': r2
                }
            }
        }, f)
    
    return clf_pipeline, reg_pipeline

if __name__ == "__main__":
    train_models()
