from unittest.mock import patch
from pathlib import Path
from model_pipeline.train import main

@patch("data_pipeline.data_fetcher.fetch_data")
def test_training_produces_artifact(mock_fetch):
    # Skip real data loading
    mock_fetch.return_value = None

    # Run train.py logic
    main()

    # Check artifact path
    assert Path("trained_models/model-latest.joblib").exists()
