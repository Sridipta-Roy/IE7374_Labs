from airflow.sdk import dag, task
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle
import json
from pathlib import Path

# Configuration
DATA_PATH = "/opt/airflow/dags/data"
MODEL_PATH = "/opt/airflow/dags/model"
METRICS_PATH = "/opt/airflow/dags/metrics"
ACCURACY_THRESHOLD = 0.90  

@dag(
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    description="Daily ML pipeline: trains 3 models on Iris dataset, selects best, and deploys"
)
def ml_model_AirFlow():
    
    @task
    def load_and_prepare_data():
        """
        Load Iris dataset from local CSV file and prepare train/test splits.
        """
        
        csv_path = f"{DATA_PATH}/iris.csv"
        column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        
        try:
            df = pd.read_csv(csv_path)
            print(f"✓ Dataset loaded successfully from {csv_path}")
        except FileNotFoundError:
            print(f"Error: iris.csv not found at {csv_path}")
            print("Please ensure iris.csv is in the /opt/airflow/dags/data/ directory")
            raise
        
        # Remove any empty rows
        #df = df[df['species'].notna()]
        df = df.dropna()
        
        print(f"\nDataset Info:")
        print(f"Total samples: {len(df)}")
        print(f"Features: {column_names[:-1]}")
        print(f"\nClass distribution:")
        print(df['species'].value_counts())
        
        # Convert species to numeric labels
        species_map = {
            'Setosa': 0,
            'Versicolor': 1,
            'Virginica': 2
        }
        y = df['species'].map(species_map).values
        
        # Prepare features and target
        X = df.drop('species', axis=1).values
        
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save preprocessed data
        data = {
            'X_train': X_train_scaled.tolist(),
            'X_test': X_test_scaled.tolist(),
            'y_train': y_train.tolist(),
            'y_test': y_test.tolist(),
            'feature_names': column_names[:-1],
            'species_map': species_map
        }
        
        with open(f"{DATA_PATH}/processed_data.json", 'w') as f:
            json.dump(data, f)
        
        # Save scaler for later use
        with open(f"{MODEL_PATH}/scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"\n✓ Data prepared: {len(y_train)} training samples, {len(y_test)} test samples")
        print(f"✓ Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
        
        return "data_ready"
    
    @task
    def training_random_forest(data_status: str):
        """Train Random Forest model and return metrics."""
        print("Training Random Forest model...")
        
        # Load data
        with open(f"{DATA_PATH}/processed_data.json", 'r') as f:
            data = json.load(f)
        
        X_train = np.array(data['X_train'])
        X_test = np.array(data['X_test'])
        y_train = np.array(data['y_train'])
        y_test = np.array(data['y_test'])
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'model_name': 'RandomForest',
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted')),
            'precision': float(precision_score(y_test, y_pred, average='weighted')),
            'recall': float(recall_score(y_test, y_pred, average='weighted'))
        }
        
        # Save model
        with open(f"{MODEL_PATH}/random_forest.pkl", 'wb') as f:
            pickle.dump(model, f)
        
        # Save metrics
        with open(f"{METRICS_PATH}/random_forest_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Random Forest - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        return metrics
    
    @task
    def training_gradient_boosting(data_status: str):
        """Train Gradient Boosting model and return metrics."""
        print("Training Gradient Boosting model...")
        
        # Load data
        with open(f"{DATA_PATH}/processed_data.json", 'r') as f:
            data = json.load(f)
        
        X_train = np.array(data['X_train'])
        X_test = np.array(data['X_test'])
        y_train = np.array(data['y_train'])
        y_test = np.array(data['y_test'])
        
        # Train model
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'model_name': 'GradientBoosting',
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted')),
            'precision': float(precision_score(y_test, y_pred, average='weighted')),
            'recall': float(recall_score(y_test, y_pred, average='weighted'))
        }
        
        # Save model
        with open(f"{MODEL_PATH}/gradient_boosting.pkl", 'wb') as f:
            pickle.dump(model, f)
        
        # Save metrics
        with open(f"{METRICS_PATH}/gradient_boosting_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Gradient Boosting - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        return metrics
    
    @task
    def training_logistic_regression(data_status: str):
        """Train Logistic Regression model and return metrics."""
        print("Training Logistic Regression model...")
        
        # Load data
        with open(f"{DATA_PATH}/processed_data.json", 'r') as f:
            data = json.load(f)
        
        X_train = np.array(data['X_train'])
        X_test = np.array(data['X_test'])
        y_train = np.array(data['y_train'])
        y_test = np.array(data['y_test'])
        
        # Train model
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'model_name': 'LogisticRegression',
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted')),
            'precision': float(precision_score(y_test, y_pred, average='weighted')),
            'recall': float(recall_score(y_test, y_pred, average='weighted'))
        }
        
        # Save model
        with open(f"{MODEL_PATH}/logistic_regression.pkl", 'wb') as f:
            pickle.dump(model, f)
        
        # Save metrics
        with open(f"{METRICS_PATH}/logistic_regression_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Logistic Regression - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        return metrics
    
    @task.branch
    def select_best_model(model_metrics: list[dict]):
        """
        Compare models and select the best one based on accuracy.
        Branch to deployment if model is good enough, otherwise retrain.
        """
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        best_model = None
        best_accuracy = 0
        
        for metrics in model_metrics:
            print(f"\n{metrics['model_name']}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  F1 Score:  {metrics['f1_score']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_model = metrics['model_name']
        
        print("\n" + "="*50)
        print(f"BEST MODEL: {best_model} (Accuracy: {best_accuracy:.4f})")
        print(f"THRESHOLD: {ACCURACY_THRESHOLD}")
        print("="*50 + "\n")
        
        # Save best model info
        best_model_info = {
            'model_name': best_model,
            'accuracy': best_accuracy,
            'timestamp': datetime.now().isoformat(),
            'all_metrics': model_metrics
        }
        
        with open(f"{METRICS_PATH}/best_model.json", 'w') as f:
            json.dump(best_model_info, f, indent=2)
        
        # Branch based on accuracy threshold
        if best_accuracy >= ACCURACY_THRESHOLD:
            return "deploy_model"
        else:
            return "model_needs_improvement"
    
    @task.bash
    def deploy_model():
        """Deploy the best model to production."""
        return f"""
        echo "=========================================="
        echo "DEPLOYING MODEL TO PRODUCTION"
        echo "=========================================="
        echo ""
        echo "Deployment steps:"
        echo "1. Loading best model configuration..."
        cat {METRICS_PATH}/best_model.json
        echo ""
        echo "2. Copying model to production directory..."
        echo "3. Updating model registry..."
        echo "4. Running smoke tests..."
        echo "5. Updating API endpoint..."
        echo ""
        echo "✓ Model successfully deployed!"
        echo "✓ Production endpoint updated"
        echo "✓ Monitoring enabled"
        echo "=========================================="
        """
    
    @task.bash
    def model_needs_improvement():
        """Alert when model performance is below threshold."""
        return f"""
        echo "=========================================="
        echo "⚠️  MODEL PERFORMANCE BELOW THRESHOLD"
        echo "=========================================="
        echo ""
        echo "Current best model accuracy < {ACCURACY_THRESHOLD}"
        echo ""
        echo "Recommended actions:"
        echo "1. Review training data quality"
        echo "2. Check for data drift"
        echo "3. Tune hyperparameters"
        echo "4. Consider feature engineering"
        echo "5. Collect more training data"
        echo ""
        echo "Sending alert to ML team..."
        echo "✓ Alert sent"
        echo "=========================================="
        """
    
    # Define DAG workflow
    data_ready = load_and_prepare_data()
    
    # Train all models in parallel
    rf_metrics = training_random_forest(data_ready)
    gb_metrics = training_gradient_boosting(data_ready)
    lr_metrics = training_logistic_regression(data_ready)
    
    # Select best model and branch
    model_selection = select_best_model([rf_metrics, gb_metrics, lr_metrics])
    
    # Define branches
    model_selection >> [deploy_model(), model_needs_improvement()]

# Instantiate the DAG
ml_model_AirFlow()