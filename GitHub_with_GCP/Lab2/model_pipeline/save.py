import os
from pathlib import Path
import shutil
from xml.parsers.expat import model
import joblib
from google.cloud import storage


ARTIFACTS_DIR = Path("trained_models")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

VERSION_FILE = Path("version.txt")

def read_version() -> int:
    if VERSION_FILE.exists():
        return int(VERSION_FILE.read_text().strip() or 0)
    return 0

def write_version(new_version: int) -> None:
    VERSION_FILE.write_text(str(new_version))   

def bump_version() -> int:
    v = read_version() + 1
    write_version(v)
    return v

# Utility to log metrics to GCS
def log_metrics_to_gcs(version, accuracy, bucket_name, log_path="metrics/metrics_log.csv"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(log_path)
    try:
        existing = blob.download_as_text()
        new_entry = f"{version},{accuracy:.4f}"
        updated = existing + new_entry
    except Exception:
        updated = "version,accuracy" + f"{version},{accuracy:.4f}"
        blob.upload_from_string(updated, content_type="text/csv")

def save_model(model, version: int) -> str:
    path = ARTIFACTS_DIR / f"model-v{version}.joblib"
    joblib.dump(model, path)
    latest = ARTIFACTS_DIR / "model-latest.joblib"
    if os.name == "nt":
        if latest.exists():
            latest.unlink()
        shutil.copy(path, latest)
    else:
        # ✅ On Linux/macOS – use symlink
        if latest.exists():
            latest.unlink()
        latest.symlink_to(path.name)
    return str(path)