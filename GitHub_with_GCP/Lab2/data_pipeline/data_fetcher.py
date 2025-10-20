import os
import io
import pandas as pd
from google.cloud import storage
from dotenv import load_dotenv

if os.path.exists(".env"):
    load_dotenv()

def fetch_data() -> pd.DataFrame:
    """Fetch CSV data from GCP bucket"""
    bucket_name = os.getenv("GCP_BUCKET")
    file_path = os.getenv("GCP_FILE_PATH")
    if not bucket_name or not file_path:
        raise ValueError("GCP_BUCKET and GCP_FILE_PATH must be set in environment.")
    client = client = storage.Client(project="github-labs-mlops")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    data = blob.download_as_text()
    df = pd.read_csv(io.StringIO(data))
    return df