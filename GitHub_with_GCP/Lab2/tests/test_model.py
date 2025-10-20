from unittest.mock import patch
from pathlib import Path
import pandas as pd
from model_pipeline.train import main

@patch("model_pipeline.train.fetch_data")  
def test_training_produces_artifact(mock_fetch):
    # Return mock data matching the expected structure from train.py
    
    mock_fetch.return_value = pd.DataFrame({
        "sepal_length": [5.1, 4.9, 4.7, 5.0, 5.4, 
                         6.2, 5.8, 6.9, 6.3, 5.9,
                         7.2, 6.5, 7.1, 6.8, 7.3],
        "sepal_width": [3.5, 3.0, 3.2, 3.6, 3.4,
                        2.9, 2.7, 3.1, 2.8, 3.0,
                        3.0, 2.9, 3.0, 2.8, 2.9],
        "petal_length": [1.4, 1.4, 1.3, 1.5, 1.7,
                         4.3, 4.1, 5.4, 4.5, 4.2,
                         5.8, 5.6, 5.9, 5.5, 6.1],
        "petal_width": [0.2, 0.2, 0.2, 0.3, 0.4,
                        1.3, 1.0, 1.5, 1.2, 1.3,
                        1.6, 1.4, 1.7, 1.5, 1.8],
        "species": ["setosa", "setosa", "setosa", "setosa", "setosa",
                    "versicolor", "versicolor", "versicolor", "versicolor", "versicolor",
                    "virginica", "virginica", "virginica", "virginica", "virginica"]
    })

    # Run train.py logic
    main()

    # Check artifact path
    assert Path("trained_models/model-latest.joblib").exists()