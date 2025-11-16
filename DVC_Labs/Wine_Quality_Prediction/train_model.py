"""
Model Training Module : Trains a Random Forest classifier on wine quality data
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import yaml
import os
import joblib
import json

def load_params(params_path='params.yaml'):
    """Load parameters from params.yaml"""
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    return params

def train_model():
    """Train the Random Forest model"""
    print("Starting model training...")
    
    # Load parameters
    params = load_params()
    n_estimators = params['model']['n_estimators']
    max_depth = params['model']['max_depth']
    random_state = params['model']['random_state']
    
    # Load feature-engineered data
    train_data = pd.read_csv('data/features/train_features.csv')
    test_data = pd.read_csv('data/features/test_features.csv')
    
    # Separate features and target
    X_train = train_data.drop('quality_label', axis=1)
    y_train = train_data['quality_label']
    X_test = test_data.drop('quality_label', axis=1)
    y_test = test_data['quality_label']
    
    print(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_metrics = {
        'accuracy': float(accuracy_score(y_train, y_train_pred)),
        'precision': float(precision_score(y_train, y_train_pred)),
        'recall': float(recall_score(y_train, y_train_pred)),
        'f1': float(f1_score(y_train, y_train_pred))
    }
    
    test_metrics = {
        'accuracy': float(accuracy_score(y_test, y_test_pred)),
        'precision': float(precision_score(y_test, y_test_pred)),
        'recall': float(recall_score(y_test, y_test_pred)),
        'f1': float(f1_score(y_test, y_test_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_test_proba))
    }
    
    print("\n=== Training Metrics ===")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n=== Test Metrics ===")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics', exist_ok=True)
    
    # Save model
    joblib.dump(model, 'models/model.pkl')
    
    # Save metrics
    with open('metrics/train_metrics.json', 'w') as f:
        json.dump(train_metrics, f, indent=4)
    
    with open('metrics/test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    print("\nModel training completed!")
    print(f"Model saved to: models/model.pkl")

if __name__ == '__main__':
    train_model()