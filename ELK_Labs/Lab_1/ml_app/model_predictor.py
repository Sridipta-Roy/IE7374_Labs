import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json
import socket
import time
from datetime import datetime

class MLModelPredictor:
    def __init__(self, logstash_host='logstash', logstash_port=5000):
        self.logstash_host = logstash_host
        self.logstash_port = logstash_port
        self.model = None
        self.class_names = None
        
    def train_model(self):
        """Train a simple iris classification model"""
        print("Training model...")
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        self.class_names = iris.target_names
        
        print("Model trained successfully!")
        return X_test, y_test
    
    def predict(self, features):
        """Make a prediction and return confidence"""
        start_time = time.time()
        
        prediction = self.model.predict([features])[0]
        probabilities = self.model.predict_proba([features])[0]
        confidence = float(max(probabilities))
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return prediction, confidence, inference_time
    
    def send_to_logstash(self, log_data):
        """Send log data to Logstash via TCP"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.logstash_host, self.logstash_port))
            
            log_json = json.dumps(log_data) + '\n'
            sock.sendall(log_json.encode('utf-8'))
            sock.close()
            
            return True
        except Exception as e:
            print(f"Error sending to Logstash: {e}")
            return False
    
    def log_prediction(self, features, predicted_label, actual_label, 
                      confidence, inference_time, model_version="v1.0"):
        """Create and send prediction log"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": "iris_classifier",
            "model_version": model_version,
            "predicted_label": int(predicted_label),
            "predicted_class": self.class_names[predicted_label],
            "actual_label": int(actual_label),
            "actual_class": self.class_names[actual_label],
            "confidence": round(confidence, 4),
            "inference_time_ms": round(inference_time, 2),
            "features": {
                "sepal_length": float(features[0]),
                "sepal_width": float(features[1]),
                "petal_length": float(features[2]),
                "petal_width": float(features[3])
            },
            "environment": "production"
        }
        
        return self.send_to_logstash(log_data)