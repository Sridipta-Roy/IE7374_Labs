from unittest.mock import patch
from pathlib import Path
import pandas as pd
from model_pipeline.train import main

@patch("model_pipeline.train.fetch_data")  # Patch where it's USED, not where it's defined
def test_training_produces_artifact(mock_fetch):
    # Return mock data matching the expected structure from train.py
    # Need at least 2 samples per class for stratified split
    mock_fetch.return_value = pd.DataFrame({
        "sepal_length": [5.1, 4.9, 6.2, 5.8, 7.2, 6.9, 5.0, 6.3, 7.1, 6.5],
        "sepal_width": [3.5, 3.0, 2.9, 2.7, 3.0, 3.1, 3.6, 2.8, 3.0, 2.9],
        "petal_length": [1.4, 1.4, 4.3, 4.1, 5.8, 5.4, 1.5, 4.5, 5.9, 5.6],
        "petal_width": [0.2, 0.2, 1.3, 1.0, 1.6, 1.5, 0.3, 1.2, 1.7, 1.4],
        "species": ["setosa", "setosa", "setosa", "versicolor", "versicolor", 
                    "versicolor", "virginica", "virginica", "virginica", "virginica"]
    })

    # Run train.py logic
    main()

    # Check artifact path
    assert Path("trained_models/model-latest.joblib").exists()