import time
import psutil
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import numpy as np

class ModelTrainer:
    def __init__(self, logger, model_type, hyperparameters):
        self.logger = logger
        self.model_type = model_type
        self.hyperparameters = hyperparameters
        self.model = None
    
    def get_model(self):
        """Initialize model based on type"""
        if self.model_type == "logistic_regression":
            return LogisticRegression(**self.hyperparameters)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(**self.hyperparameters)
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(**self.hyperparameters)
        elif self.model_type == "svm":
            return SVC(**self.hyperparameters, probability=True)
        else:
            return LogisticRegression(**self.hyperparameters)
    
    def train(self, X_train, y_train):
        """Train model"""
        self.logger.log_pipeline_event(
            stage="training",
            event_type="start",
            data={
                "model_type": self.model_type,
                "hyperparameters": self.hyperparameters,
                "message": f"Starting training for {self.model_type}"
            }
        )
        
        start_time = time.time()
        cpu_before = psutil.cpu_percent(interval=0.1)
        memory_before = psutil.virtual_memory().percent
        
        try:
            self.model = self.get_model()
            
            # Training
            self.model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            cpu_after = psutil.cpu_percent(interval=0.1)
            memory_after = psutil.virtual_memory().percent
            
            # Get training accuracy
            train_score = self.model.score(X_train, y_train)
            
            self.logger.log_pipeline_event(
                stage="training",
                event_type="complete",
                data={
                    "model_type": self.model_type,
                    "training_time_seconds": round(training_time, 3),
                    "train_accuracy": round(train_score, 4),
                    "cpu_percent": round((cpu_before + cpu_after) / 2, 2),
                    "memory_percent": round((memory_before + memory_after) / 2, 2),
                    "n_samples": len(X_train),
                    "message": "Training completed successfully"
                }
            )
            
            return self.model
            
        except Exception as e:
            self.logger.log_error("training", "Failed to train model", e)
            raise