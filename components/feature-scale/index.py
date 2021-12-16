import pickle
from pathlib import Path
from pandas import read_csv
from json import dumps, dump
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler
parser = ArgumentParser()

parser.add_argument('--x-train', type=str)
parser.add_argument('--X-train', type=str)
parser.add_argument('--x-test', type=str)
parser.add_argument('--X-test', type=str)
parser.add_argument('--standard-scaler', type=str)

args = parser.parse_args()

x_train = read_csv(args.x_train)
x_test = read_csv(args.x_test)

standard_scaler = StandardScaler()
x_train = standard_scaler.fit_transform(x_train)
x_test = standard_scaler.transform(x_test)

Path(args.X_train).parent.mkdir(exist_ok=True, parents=True)
with open(args.X_train, 'w') as f:
    dump(dumps(x_test.tolist()), f)

Path(args.X_test).parent.mkdir(exist_ok=True, parents=True)
with (args.X_test, 'w') as f:
    dump(dumps(x_test.tolist()), f)

Path(args.standard_scaler).parent.mkdir(exist_ok=True, parents=True)
with open(args.standard_scaler, 'w') as f:
    pickle.dump(standard_scaler, f)