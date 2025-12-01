import time
from datetime import datetime
from src.logger import MLOpsLogger
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.model_trainer import ModelTrainer
from src.evaluator import Evaluator

class MLOpsPipeline:
    def __init__(self, experiment_config):
        self.experiment_name = experiment_config['name']
        self.model_type = experiment_config['model_type']
        self.hyperparameters = experiment_config['hyperparameters']
        self.description = experiment_config.get('description', '')
        
        # Generate experiment ID
        self.experiment_id = f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize logger
        self.logger = MLOpsLogger(self.experiment_id)
    
    def run(self, dataset='breast_cancer'):
        """Run complete ML pipeline"""
        pipeline_start_time = time.time()
        
        self.logger.log_pipeline_event(
            stage="pipeline",
            event_type="start",
            data={
                "experiment_name": self.experiment_name,
                "model_type": self.model_type,
                "description": self.description,
                "dataset": dataset,
                "hyperparameters": self.hyperparameters,
                "message": f"Starting pipeline for {self.experiment_name}"
            }
        )
        
        try:
            # Stage 1: Data Loading
            data_loader = DataLoader(self.logger, dataset)
            X_train, X_test, y_train, y_test, feature_names = data_loader.load_data()
            
            # Stage 2: Preprocessing
            preprocessor = Preprocessor(self.logger)
            X_train_scaled, X_test_scaled = preprocessor.preprocess(X_train, X_test)
            
            # Stage 3: Model Training
            trainer = ModelTrainer(self.logger, self.model_type, self.hyperparameters)
            model = trainer.train(X_train_scaled, y_train)
            
            # Stage 4: Evaluation
            evaluator = Evaluator(self.logger)
            metrics = evaluator.evaluate(
                model, X_train_scaled, X_test_scaled, 
                y_train, y_test, self.model_type
            )
            
            # Calculate total pipeline time
            total_time = time.time() - pipeline_start_time
            
            # Log final metrics
            final_metrics = {
                "experiment_name": self.experiment_name,
                "model_type": self.model_type,
                "dataset": dataset,
                "hyperparameters": self.hyperparameters,
                "total_training_time": round(total_time, 3),
                **metrics
            }
            
            self.logger.log_metrics(final_metrics)
            
            self.logger.log_pipeline_event(
                stage="pipeline",
                event_type="complete",
                data={
                    "experiment_name": self.experiment_name,
                    "total_time_seconds": round(total_time, 3),
                    "test_accuracy": metrics['test_accuracy'],
                    "f1_score": metrics['f1_score'],
                    "message": f"Pipeline completed successfully for {self.experiment_name}"
                }
            )
            
            return final_metrics
            
        except Exception as e:
            self.logger.log_error("pipeline", f"Pipeline failed for {self.experiment_name}", e)
            
            self.logger.log_pipeline_event(
                stage="pipeline",
                event_type="failed",
                data={
                    "experiment_name": self.experiment_name,
                    "error": str(e),
                    "message": f"Pipeline failed for {self.experiment_name}"
                }
            )
            
            raise