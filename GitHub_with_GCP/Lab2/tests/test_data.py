from data_pipeline.data_fetcher import fetch_data


def test_fetch_data_shape():
    df = fetch_data()
    assert df.shape[0] > 0
    assert "species" in df.columns