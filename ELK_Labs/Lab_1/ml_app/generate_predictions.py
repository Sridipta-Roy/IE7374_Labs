import time
import numpy as np
from model_predictor import MLModelPredictor

def main():
    print("Starting ML Prediction Service...")
    
    # Wait for Logstash to be ready
    print("Waiting for Logstash to be ready...")
    time.sleep(30)
    
    predictor = MLModelPredictor()
    
    # Train the model
    X_test, y_test = predictor.train_model()
    
    print("\nStarting to generate predictions...")
    print("Press Ctrl+C to stop\n")
    
    prediction_count = 0
    
    try:
        # Generate predictions continuously
        for i in range(len(X_test)):
            features = X_test[i]
            actual_label = y_test[i]
            
            # Make prediction
            predicted_label, confidence, inference_time = predictor.predict(features)
            
            # Log to Logstash
            success = predictor.log_prediction(
                features, predicted_label, actual_label, 
                confidence, inference_time
            )
            
            prediction_count += 1
            
            if success:
                status = "✓" if predicted_label == actual_label else "✗"
                print(f"{status} Prediction {prediction_count}: "
                      f"Predicted={predictor.class_names[predicted_label]}, "
                      f"Actual={predictor.class_names[actual_label]}, "
                      f"Confidence={confidence:.2f}, "
                      f"Time={inference_time:.2f}ms")
            
            # Wait between predictions
            time.sleep(2)
            
            # Loop back to beginning when done with test set
            if i == len(X_test) - 1:
                print("\n--- Completed one cycle, restarting... ---\n")
                i = 0
                
    except KeyboardInterrupt:
        print(f"\n\nStopped after {prediction_count} predictions")

if __name__ == "__main__":
    main()