from ucimlrepo import fetch_ucirepo
from training import create_forest
import numpy as np

# fetch dataset
car_evaluation = fetch_ucirepo(id=19)

# data (as pandas dataframes)
X = car_evaluation.data.features
Y = car_evaluation.data.targets

targets = [Y.iloc[i, 0] for i in range(0, 1728)]

data = []
for i in range(0, 6):
    column = []
    for k in range(0, 1728):
        column.append(X.iloc[k, i])
    data.append(column)

create_forest(targets, data)

print(Y)
print(X.iloc[0, 0])