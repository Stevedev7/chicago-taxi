from numpy import array
from pickle import dump
from pathlib import Path
from pandas import read_csv
from argparse import ArgumentParser
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import MeanAbsoluteError
from json import load as json_load, loads as json_loads


parser = ArgumentParser()

parser.add_argument('--x-train', type=str)
parser.add_argument('--y-train', type=str)
parser.add_argument('--model', type=str)

args = parser.parse_args()

with open(args.x_train, 'r') as f:
    x_train = array(json_loads(json_load(f)))

y_train = read_csv(args.y_train).values

model = Sequential()

model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='linear'))

model.compile(
      loss='mse',
      optimizer=SGD(learning_rate=0.001),
      metrics=[MeanAbsoluteError()]
)

model.fit(x_train, y_train, batch_size=64, epochs=200)

Path(args.model).parent.mkdir(exist_ok=True, parents=True)

with open (args.model, 'wb') as f:
    dump(model, f)