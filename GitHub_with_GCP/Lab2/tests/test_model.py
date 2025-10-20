from pathlib import Path
from unittest.mock import patch
from model_pipeline.train import main

@patch("data_pipeline.data_fetcher.fetch_data")
def test_training_produces_artifact(mock_fetch):
    # Mock empty dataset to pass into training
    mock_fetch.return_value = None

    main()  # Run training pipeline directly, no subprocess
    assert Path("trained_models/model-latest.joblib").exists()
