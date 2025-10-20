import pandas as pd
from unittest.mock import patch
from data_pipeline.data_fetcher import fetch_data

@patch("data_pipeline.data_fetcher.storage.Client")
@patch("data_pipeline.data_fetcher.fetch_data")
def test_fetch_data_from_gcp(mock_fetch, mock_client):
    # Simulate final DataFrame result
    mock_fetch.return_value = pd.DataFrame({
        "col1": [1, 2],
        "col2": [3, 4]
    })

    df = fetch_data()
    assert not df.empty
    assert list(df.columns) == ["col1", "col2"]
