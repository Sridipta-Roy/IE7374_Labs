import pandas as pd
from unittest.mock import patch
from data_pipeline.data_fetcher import fetch_data

@patch("data_pipeline.data_fetcher.fetch_data")
def test_fetch_data_shape(mock_fetch):
    # Provide a fake DataFrame instead of real GCP/data call
    mock_fetch.return_value = pd.DataFrame({
        "species": ["setosa", "virginica"],
        "value": [1, 2]
    })

    df = fetch_data()
    assert df.shape[0] > 0
    assert "species" in df.columns
