import os
from pathlib import Path
from typing import List


import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()


PORT = int(os.getenv("PORT", 8080))
MODEL_LOCAL_PATH = Path(os.getenv("MODEL_LOCAL_PATH", "trained_models/model-latest.joblib"))


app = FastAPI(title="MLOps Lab - Cloud Run API", version="1.0")


class PredictRequest(BaseModel):
    # Expect 4 iris features per row
    features: List[List[float]]


class PredictResponse(BaseModel):
    predictions: List[str]


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_LOCAL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_LOCAL_PATH}")
    model = joblib.load(MODEL_LOCAL_PATH)


@app.get("/")
async def root():
    return {"status": "ok", "model_path": str(MODEL_LOCAL_PATH)}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    X = np.array(req.features)
    preds = model.predict(X)
    print("DEBUG preds =", preds, type(preds))
    return PredictResponse(predictions=[str(p) for p in preds])