"""
Data Preprocessing Module: Loads and preprocesses the Wine Quality dataset
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import os

def load_params(params_path='params.yaml'):
    """Load parameters from params.yaml"""
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    return params

def preprocess_data():
    """Load and preprocess the Wine Quality dataset"""
    print("Loading Wine Quality dataset...")
    
    # Load parameters
    params = load_params()
    test_size = params['data']['test_size']
    random_state = params['data']['random_state']
    
    # Load dataset from UCI ML Repository
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    # Read the dataset
    df = pd.read_csv(url, sep=';')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    print(f"\nQuality distribution:\n{df['quality'].value_counts().sort_index()}")
    
    # Convert to binary classification: good wine (quality >= 7) vs not good
    df['quality_label'] = (df['quality'] >= 7).astype(int)
    
    # Drop original quality column
    df = df.drop('quality', axis=1)
    
    # Create data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Save raw data
    df.to_csv('data/raw/wine_data.csv', index=False)
    
    # Split data
    X = df.drop('quality_label', axis=1)
    y = df['quality_label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Save processed data
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_csv('data/processed/train.csv', index=False)
    test_data.to_csv('data/processed/test.csv', index=False)
    
    print(f"\nTraining set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")
    print(f"Positive class ratio (train): {y_train.mean():.2%}")
    print("Data preprocessing completed!")

if __name__ == '__main__':
    preprocess_data()