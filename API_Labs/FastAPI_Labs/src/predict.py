import joblib
import numpy as np

def predict_data(X):
    """
    Predict the class labels for the input data with probabilities.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        dict: Dictionary containing prediction, confidence, and all probabilities.
    """
    model = joblib.load("../model/iris_model.pkl")
    
    # Get the predicted class
    y_pred = model.predict(X)
    
    # Get prediction probabilities
    y_proba = model.predict_proba(X)
    
    # Class names (iris species)
    class_names = ['setosa', 'versicolor', 'virginica']
    
    # For single prediction (assuming X is a single sample)
    if len(X) == 1:
        predicted_class = class_names[y_pred[0]]
        probabilities = y_proba[0]
        confidence = max(probabilities)
        
        # Create probability dictionary
        all_probabilities = {
            class_names[i]: round(prob, 2) 
            for i, prob in enumerate(probabilities)
        }
        
        return {
            "prediction": predicted_class,
            "confidence": round(confidence, 2),
            "all_probabilities": all_probabilities
        }
    
    # For batch predictions
    else:
        results = []
        for i in range(len(X)):
            predicted_class = class_names[y_pred[i]]
            probabilities = y_proba[i]
            confidence = max(probabilities)
            
            all_probabilities = {
                class_names[j]: round(prob, 2) 
                for j, prob in enumerate(probabilities)
            }
            
            results.append({
                "prediction": predicted_class,
                "confidence": round(confidence, 2),
                "all_probabilities": all_probabilities
            })
        
        return results
