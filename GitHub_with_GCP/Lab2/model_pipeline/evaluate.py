from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from google.cloud import storage
import os
import io
import pandas as pd
from dotenv import load_dotenv

if os.path.exists(".env"):
    load_dotenv()

MODEL_PATH = Path("trained_models/model-latest.joblib")

def evaluate(path: Path = MODEL_PATH) -> float:
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    model = joblib.load(path)
    
    bucket_name = os.getenv("GCP_BUCKET")
    file_path = os.getenv("GCP_TEST_FILE_PATH")
    if not bucket_name or not file_path:
        raise ValueError("GCP_BUCKET and GCP_TEST_FILE_PATH must be set in environment.")
    
    client = storage.Client(project=os.getenv("GCP_PROJECT_ID"))
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    data = blob.download_as_text()
    iris = pd.read_csv(io.StringIO(data))
    
    X = iris.drop(columns=["species"]) # quick sanity eval on full dataset
    y = iris["species"]
    preds = model.predict(X)
    return accuracy_score(y, preds)




if __name__ == "__main__":
    
    acc = evaluate()
    print(f"accuracy={acc:.4f}")
    # ✅ Accuracy threshold check
    threshold = 0.88
    if acc < threshold:
        raise ValueError(f"Model accuracy {acc:.4f} is below threshold {threshold}. Stopping pipeline.")
   