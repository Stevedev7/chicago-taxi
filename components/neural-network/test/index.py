from pickle import load
from numpy import array
from pandas import read_csv
from argparse import ArgumentParser
from json import load as json_load, loads
from sklearn.metrics import mean_absolute_error

parser = ArgumentParser()

parser.add_argument('--x-test', type=str)
parser.add_argument('--y-test', type=str)
parser.add_argument('--model', type=str)

args = parser.parse_args()

# with open(args.x_test, 'r') as f:
#     x_test = array(loads(json_load(f)))

with open(args.x_test, 'r') as f:
    x = array(loads(json_load(f)))

y_test = read_csv(args.y_test).values

with open(args.model, 'rb') as f:
    model = load(f)


y_pred = model.predict(x)

mse = mean_absolute_error(y_test, y_pred)

print(mse)