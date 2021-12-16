import pandas as pd
from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--dataset', type=str)
parser.add_argument('--X', type=str)
parser.add_argument('--y', type=str)

args = parser.parse_args()

df = pd.read_csv(args.dataset)

df.drop(['unique_key', 'taxi_id', 'trip_start_timestamp', 'trip_end_timestamp', 'pickup_census_tract',
       'dropoff_census_tract', 'pickup_community_area',
       'dropoff_community_area', 'pickup_latitude',
       'pickup_longitude', 'pickup_location', 'dropoff_latitude',
       'dropoff_longitude', 'dropoff_location'], axis=1, inplace=True)

df.dropna(inplace=True, axis=0)

X = df.drop('trip_total', axis=1)
y = df.trip_total

args = parser.parse_args()

for key in args.__dict__:
    if key != 'dataset':
        Path(args.__dict__[key]).parent.mkdir(exist_ok=True, parents=True)

X.to_csv(args.X, index=False)
y.to_csv(args.y, index=False)