from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from predict import predict_data
import numpy as np

app = FastAPI(
    title="Iris Classification API",
    description="A simple machine learning API for iris species prediction",
    version="1.0.0"
)

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisResponse(BaseModel):
    prediction: str
    confidence: float
    all_probabilities: Dict[str, float]

class BatchIrisData(BaseModel):
    samples: List[IrisData]

class BatchIrisResponse(BaseModel):
    predictions: List[IrisResponse]

class HealthResponse(BaseModel):
    status: str
    message: str

@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint to verify API is running.
    """
    return HealthResponse(
        status="healthy",
        message="Iris Classification API is running"
    )

@app.get("/", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def root():
    """
    Root endpoint - redirects to health check.
    """
    return HealthResponse(
        status="healthy",
        message="Iris Classification API is running"
    )

@app.post("/predict", response_model=IrisResponse, status_code=status.HTTP_200_OK)
async def predict_iris(iris_features: IrisData):
    """
    Predict iris species for a single sample.
    
    Args:
        iris_features: IrisData containing sepal and petal measurements
        
    Returns:
        IrisResponse with prediction, confidence, and all probabilities
    """
    try:
        # Convert input to numpy array format expected by predict_data
        features = np.array([[
            iris_features.sepal_length, 
            iris_features.sepal_width,
            iris_features.petal_length, 
            iris_features.petal_width
        ]])
        
        # Get prediction from model
        prediction_result = predict_data(features)
        
        return IrisResponse(
            prediction=prediction_result["prediction"],
            confidence=prediction_result["confidence"],
            all_probabilities=prediction_result["all_probabilities"]
        )
   
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchIrisResponse, status_code=status.HTTP_200_OK)
async def predict_iris_batch(batch_data: BatchIrisData):
    """
    Predict iris species for multiple samples.
    
    Args:
        batch_data: BatchIrisData containing list of iris measurements
        
    Returns:
        BatchIrisResponse with predictions for all samples
    """
    try:
        # Convert batch input to numpy array
        features_list = []
        for sample in batch_data.samples:
            features_list.append([
                sample.sepal_length,
                sample.sepal_width,
                sample.petal_length,
                sample.petal_width
            ])
        
        features = np.array(features_list)
        
        # Get batch predictions from model
        batch_results = predict_data(features)
        
        # Convert results to response format
        predictions = []
        for result in batch_results:
            predictions.append(IrisResponse(
                prediction=result["prediction"],
                confidence=result["confidence"],
                all_probabilities=result["all_probabilities"]
            ))
        
        return BatchIrisResponse(predictions=predictions)
   
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Batch prediction failed: {str(e)}"
        )

# Additional endpoint to get model info
@app.get("/model/info", status_code=status.HTTP_200_OK)
async def get_model_info():
    """
    Get information about the trained model.
    """
    return {
        "model_type": "Random Forest Classifier",
        "features": [
            "sepal_length",
            "sepal_width", 
            "petal_length",
            "petal_width"
        ],
        "classes": ["setosa", "versicolor", "virginica"],
        "description": "Iris species classification model trained on the famous Iris dataset"
    }
    


    
