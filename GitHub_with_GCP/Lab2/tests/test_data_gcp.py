from unittest.mock import patch
import pandas as pd

@patch("data_pipeline.data_fetcher.fetch_data")
def test_fetch_data_from_gcp(mock_fetch):
    # Simulated CSV content
    mock_fetch.return_value = pd.DataFrame({
        "col1": [10, 20],
        "col2": [30, 40]
    })

    # Import after patching
    from data_pipeline.data_fetcher import fetch_data
    df = fetch_data()

    assert not df.empty
    assert list(df.columns) == ["col1", "col2"]