from pathlib import Path
import subprocess


def test_training_produces_artifact(tmp_path: Path):
    # run training; artifact should be created
    proc = subprocess.run(["python", "model_pipeline/train.py"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert Path("trained_models/model-latest.joblib").exists()