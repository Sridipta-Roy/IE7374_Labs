import os
from unittest.mock import patch, MagicMock
from data_pipeline.data_fetcher import fetch_data


@patch("data_pipeline.data_fetcher.storage.Client")
def test_fetch_data_from_gcp(mock_client):
    # Mock bucket and blob
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.download_as_text.return_value = "col1,col2"
    mock_bucket.blob.return_value = mock_blob
    mock_client.return_value.bucket.return_value = mock_bucket


    os.environ["GCP_BUCKET"] = "test-bucket"
    os.environ["GCP_FILE_PATH"] = "test.csv"


    df = fetch_data()
    assert not df.empty
    assert list(df.columns) == ["col1", "col2"]