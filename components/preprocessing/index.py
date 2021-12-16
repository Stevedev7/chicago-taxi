import pandas as pd
from pickle import dump
from pathlib import Path
from argparse import ArgumentParser
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

parser = ArgumentParser()

parser.add_argument('--X', type=str)
parser.add_argument('--y', type=str)

parser.add_argument('--x-train', type=str)
parser.add_argument('--y-train', type=str)
parser.add_argument('--y-test', type=str)
parser.add_argument('--x-test', type=str)
parser.add_argument('--column-transformer', type=str)

args = parser.parse_args()

X = pd.read_csv(args.X)
y = pd.read_csv(args.y)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [6, 7])], remainder='passthrough')
X = ct.fit_transform(X.values)


x_train, x_test, y_train, y_test = train_test_split(X, y.values, test_size=0.2)

Path(args.x_train).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(x_train.todense()).to_csv(args.x_train, index=False)

Path(args.y_train).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(y_train).to_csv(args.y_train, index=False)

Path(args.x_test).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(x_test.todense()).to_csv(args.x_test, index=False)

Path(args.y_test).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(y_test).to_csv(args.y_test, index=False)

Path(args.column_transformer).parent.mkdir(exist_ok=True, parents=True)
with open(args.column_transformer, 'wb') as f:
    dump(ct, f)