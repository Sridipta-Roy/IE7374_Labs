"""
Feature Engineering Module : Applies feature scaling and creates additional features
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import yaml
import os
import joblib

def load_params(params_path='params.yaml'):
    """Load parameters from params.yaml"""
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    return params

def engineer_features():
    """Apply feature scaling and engineering"""
    print("Starting feature engineering...")
    
    # Load parameters
    params = load_params()
    
    # Load processed data
    train_data = pd.read_csv('data/processed/train.csv')
    test_data = pd.read_csv('data/processed/test.csv')
    
    # Separate features and target
    X_train = train_data.drop('quality_label', axis=1)
    y_train = train_data['quality_label']
    X_test = test_data.drop('quality_label', axis=1)
    y_test = test_data['quality_label']
    
    print(f"Original features: {X_train.shape[1]}")
    
    # Create interaction features
    X_train['alcohol_sulphates'] = X_train['alcohol'] * X_train['sulphates']
    X_test['alcohol_sulphates'] = X_test['alcohol'] * X_test['sulphates']
    
    X_train['acid_ratio'] = X_train['volatile acidity'] / (X_train['fixed acidity'] + 1e-5)
    X_test['acid_ratio'] = X_test['volatile acidity'] / (X_test['fixed acidity'] + 1e-5)
    
    print(f"Features after engineering: {X_train.shape[1]}")
    
    # Apply scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Create features directory
    os.makedirs('data/features', exist_ok=True)
    os.makedirs('models/artifacts', exist_ok=True)
    
    # Save scaled data
    train_features = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
    test_features = pd.concat([X_test_scaled, y_test.reset_index(drop=True)], axis=1)
    
    train_features.to_csv('data/features/train_features.csv', index=False)
    test_features.to_csv('data/features/test_features.csv', index=False)
    
    # Save scaler
    joblib.dump(scaler, 'models/artifacts/scaler.pkl')
    
    print("Feature engineering completed!")
    print(f"Saved train features: {train_features.shape}")
    print(f"Saved test features: {test_features.shape}")

if __name__ == '__main__':
    engineer_features()