from unittest.mock import patch
import pandas as pd

@patch("data_pipeline.data_fetcher.fetch_data")
def test_fetch_data_shape(mock_fetch):
    # Return mock data
    mock_fetch.return_value = pd.DataFrame({
        "species": ["setosa", "versicolor"],
        "value": [1, 2]
    })

    # Import after patching
    from data_pipeline.data_fetcher import fetch_data
    df = fetch_data()

    assert df.shape[0] > 0
    assert "species" in df.columns