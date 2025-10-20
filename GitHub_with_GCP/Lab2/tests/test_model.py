from unittest.mock import patch
from pathlib import Path
import pandas as pd
from model_pipeline.train import main

@patch("model_pipeline.train.fetch_data")
def test_training_produces_artifact(mock_fetch):
    # Return mock data matching the expected structure from train.py
    mock_fetch.return_value = pd.DataFrame({
        "sepal_length": [5.1, 4.9, 6.2, 5.8, 7.2],
        "sepal_width": [3.5, 3.0, 2.9, 2.7, 3.0],
        "petal_length": [1.4, 1.4, 4.3, 4.1, 5.8],
        "petal_width": [0.2, 0.2, 1.3, 1.0, 1.6],
        "species": ["setosa", "setosa", "versicolor", "versicolor", "virginica"]
    })

    # Run train.py logic
    main()

    # Check artifact path
    assert Path("trained_models/model-latest.joblib").exists()