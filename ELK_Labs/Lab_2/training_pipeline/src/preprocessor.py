import time
import psutil
from sklearn.preprocessing import StandardScaler
import numpy as np

class Preprocessor:
    def __init__(self, logger):
        self.logger = logger
        self.scaler = StandardScaler()
    
    def preprocess(self, X_train, X_test):
        """Preprocess data (scaling)"""
        self.logger.log_pipeline_event(
            stage="preprocessing",
            event_type="start",
            data={"message": "Starting data preprocessing"}
        )
        
        start_time = time.time()
        cpu_before = psutil.cpu_percent(interval=0.1)
        memory_before = psutil.virtual_memory().percent
        
        try:
            # Fit and transform training data
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Transform test data
            X_test_scaled = self.scaler.transform(X_test)
            
            preprocess_time = time.time() - start_time
            cpu_after = psutil.cpu_percent(interval=0.1)
            memory_after = psutil.virtual_memory().percent
            
            self.logger.log_pipeline_event(
                stage="preprocessing",
                event_type="complete",
                data={
                    "preprocessing_time_seconds": round(preprocess_time, 3),
                    "train_mean": round(float(np.mean(X_train_scaled)), 4),
                    "train_std": round(float(np.std(X_train_scaled)), 4),
                    "cpu_percent": round((cpu_before + cpu_after) / 2, 2),
                    "memory_percent": round((memory_before + memory_after) / 2, 2),
                    "message": "Preprocessing completed"
                }
            )
            
            return X_train_scaled, X_test_scaled
            
        except Exception as e:
            self.logger.log_error("preprocessing", "Failed to preprocess data", e)
            raise