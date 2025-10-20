from google.cloud import storage

# client = client = storage.Client(project="github-labs-mlops")
# buckets = list(client.list_buckets())
# print(buckets)
from data_pipeline.data_fetcher import fetch_data
print(fetch_data().head())