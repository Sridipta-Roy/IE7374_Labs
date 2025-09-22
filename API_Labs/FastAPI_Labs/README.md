# Iris Classification API

A simple machine learning API built with FastAPI that serves a Random Forest classifier trained on the Iris dataset for flower species prediction.

## 📋 Overview

This project demonstrates how to:
- Train a Random Forest classifier on the Iris dataset
- Save and load trained models
- Serve ML models via REST API using FastAPI
- Make predictions through HTTP requests

## 🚀 Features

- **Machine Learning**: Random Forest classifier for iris species prediction
- **REST API**: FastAPI-based endpoints for model inference
- **Model Persistence**: Trained model saved and loaded using joblib
- **Interactive Documentation**: Automatic API documentation with Swagger UI
- **Input Validation**: Pydantic models for request/response validation

## 📁 Project Structure

```
api_labs
└── fastapi_labs
    ├── assets/
    ├── fastapi_lab1_env/
    ├── model/
    │   └── iris_model.pkl
    ├── src/
    │   ├── __init__.py
    │   ├── data.py
    │   ├── main.py
    │   ├── predict.py
    │   └── train.py
    ├── README.md
    └── requirements.txt
```

## 📦 Dependencies

```txt
fastapi==0.104.1
uvicorn==0.24.0
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.24.3
joblib==1.3.2
pydantic==2.5.0
```

## 🏃‍♂️ Usage

### 1. Train the Model

```bash
cd src
python train.py
```

This will:
- Load the Iris dataset
- Split data into training/testing sets
- Train a Random Forest classifier
- Save the model to `model/iris_model.pkl`

### 2. Start the API Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

### 3. Access Interactive Documentation

- **Swagger UI**: `http://localhost:8000/docs`


## 🔗 API Endpoints

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "OK",
  "message": "Iris Classification API is running"
}
```

### Make Prediction
```http
POST /predict
```

**Request Body:**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response:**
```json
{
  "prediction": "setosa",
  "probability": 1,
  "class_probabilities": {
    "setosa": 1,
    "versicolor": 0,
    "virginica": 0
  }
}
```

### Batch Prediction
```http
POST /predict/batch
```

**Request Body:**
```json
{
  "samples": [
    {
      "sepal_length": 5,
      "sepal_width": 3.3,
      "petal_length": 1.4,
      "petal_width": 0.2
    },
        {
      "sepal_length": 5.7,
      "sepal_width": 2.8,
      "petal_length": 4.5,
      "petal_width": 1.3
    },
    {
      "sepal_length": 5.6,
      "sepal_width": 2.8,
      "petal_length": 4.9,
      "petal_width": 2
    }
  ]
}
```
**Response:**
```json
{
  "predictions": [
    {
      "prediction": "setosa",
      "probability": 1,
      "class_probabilities": {
        "setosa": 1,
        "versicolor": 0,
        "virginica": 0
      }
    },
    {
      "prediction": "versicolor",
      "probability": 1,
      "class_probabilities": {
        "setosa": 0,
        "versicolor": 1,
        "virginica": 0
      }
    },
    {
      "prediction": "virginica",
      "probability": 0.95,
      "class_probabilities": {
        "setosa": 0,
        "versicolor": 0.05,
        "virginica": 0.95
      }
    }
  ]
}
```

## 🧪 Testing the API

### Using curl

```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

### Using Python requests

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
)
print(response.json())
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.