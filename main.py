from ucimlrepo import fetch_ucirepo
from training import create_forest
import numpy as np

SPLIT = 1000
TREES = 100

# fetch dataset
car_evaluation = fetch_ucirepo(id=19)

# data (as pandas dataframes)
X = car_evaluation.data.features
Y = car_evaluation.data.targets

targets = [Y.iloc[i, 0] for i in range(0, 1728)]

data = []
for i in range(0, 1728):
    row = []
    for k in range(0, 6):
        row.append(X.iloc[i, k])
    data.append(row)

if SPLIT > len(data):
    raise ValueError

training_data = data[SPLIT:]
testing_data = data[:SPLIT]
training_targets = targets[SPLIT:]
testing_targets = targets[:SPLIT]

forest = create_forest(training_data, training_targets, TREES)
