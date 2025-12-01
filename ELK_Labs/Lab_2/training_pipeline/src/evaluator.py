import time
import psutil
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix)
import numpy as np

class Evaluator:
    def __init__(self, logger):
        self.logger = logger
    
    def evaluate(self, model, X_train, X_test, y_train, y_test, model_type):
        """Evaluate model performance"""
        self.logger.log_pipeline_event(
            stage="evaluation",
            event_type="start",
            data={"message": "Starting model evaluation"}
        )
        
        start_time = time.time()
        cpu_before = psutil.cpu_percent(interval=0.1)
        memory_before = psutil.virtual_memory().percent
        
        try:
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Get probabilities for ROC-AUC (if available)
            try:
                y_test_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None
            except:
                y_test_proba = None
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            # For multiclass, use weighted average
            average_type = 'binary' if len(np.unique(y_test)) == 2 else 'weighted'
            
            precision = precision_score(y_test, y_test_pred, average=average_type, zero_division=0)
            recall = recall_score(y_test, y_test_pred, average=average_type, zero_division=0)
            f1 = f1_score(y_test, y_test_pred, average=average_type, zero_division=0)
            
            # ROC-AUC only for binary classification
            roc_auc = None
            if y_test_proba is not None and len(np.unique(y_test)) == 2:
                roc_auc = roc_auc_score(y_test, y_test_proba)
            
            conf_matrix = confusion_matrix(y_test, y_test_pred)
            
            eval_time = time.time() - start_time
            cpu_after = psutil.cpu_percent(interval=0.1)
            memory_after = psutil.virtual_memory().percent
            
            metrics = {
                "model_type": model_type,
                "train_accuracy": round(train_accuracy, 4),
                "test_accuracy": round(test_accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(roc_auc, 4) if roc_auc else None,
                "confusion_matrix": conf_matrix.tolist(),
                "evaluation_time_seconds": round(eval_time, 3),
                "cpu_percent": round((cpu_before + cpu_after) / 2, 2),
                "memory_percent": round((memory_before + memory_after) / 2, 2),
            }
            
            self.logger.log_pipeline_event(
                stage="evaluation",
                event_type="complete",
                data={
                    **metrics,
                    "message": "Evaluation completed successfully"
                }
            )
            
            return metrics
            
        except Exception as e:
            self.logger.log_error("evaluation", "Failed to evaluate model", e)
            raise