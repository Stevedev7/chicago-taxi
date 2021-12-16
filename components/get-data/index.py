from pathlib import Path
from google.cloud import bigquery
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)

args = parser.parse_args()

client = bigquery.Client()

query_job = client.query(
    """
    SELECT * FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips` LIMIT 20000
"""
)

results = query_job.result()

Path(args.dataset).parent.mkdir(exist_ok=True, parents=True)
results.to_dataframe().to_csv(args.dataset, index=False)