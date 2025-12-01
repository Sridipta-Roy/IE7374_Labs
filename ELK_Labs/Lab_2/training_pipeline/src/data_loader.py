import time
import psutil
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split
import numpy as np

class DataLoader:
    def __init__(self, logger, dataset_name='breast_cancer'):
        self.logger = logger
        self.dataset_name = dataset_name
    
    def load_data(self):
        """Load and split dataset"""
        self.logger.log_pipeline_event(
            stage="data_loading",
            event_type="start",
            data={
                "dataset": self.dataset_name,
                "message": f"Loading {self.dataset_name} dataset"
            }
        )
        
        start_time = time.time()
        cpu_before = psutil.cpu_percent(interval=0.1)
        memory_before = psutil.virtual_memory().percent
        
        try:
            # Load dataset based on name
            if self.dataset_name == 'breast_cancer':
                data = load_breast_cancer()
            elif self.dataset_name == 'wine':
                data = load_wine()
            elif self.dataset_name == 'digits':
                data = load_digits()
            else:
                data = load_breast_cancer()
            
            X, y = data.data, data.target
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            load_time = time.time() - start_time
            cpu_after = psutil.cpu_percent(interval=0.1)
            memory_after = psutil.virtual_memory().percent
            
            self.logger.log_pipeline_event(
                stage="data_loading",
                event_type="complete",
                data={
                    "dataset": self.dataset_name,
                    "n_samples": len(X),
                    "n_features": X.shape[1],
                    "n_classes": len(np.unique(y)),
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "load_time_seconds": round(load_time, 3),
                    "cpu_percent": round((cpu_before + cpu_after) / 2, 2),
                    "memory_percent": round((memory_before + memory_after) / 2, 2),
                    "message": "Data loaded successfully"
                }
            )
            
            return X_train, X_test, y_train, y_test, data.feature_names
            
        except Exception as e:
            self.logger.log_error("data_loading", "Failed to load data", e)
            raise