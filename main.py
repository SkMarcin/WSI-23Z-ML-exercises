from ucimlrepo import fetch_ucirepo
from training import create_forest, get_forest_classifier, get_classification_dict, get_accuracy, get_recall, get_precision
import numpy as np

SPLIT = -1000
TREES = 300
CLASSES = ["unacc", "acc", "good", "vgood"]

# # fetch dataset
# car_evaluation = fetch_ucirepo(id=19)

# # data (as pandas dataframes)
# X = car_evaluation.data.features
# Y = car_evaluation.data.targets

# targets = [Y.iloc[i, 0] for i in range(0, 1728)]

# data = []
# for i in range(0, 1728):
#     row = []
#     for k in range(0, 6):
#         row.append(X.iloc[i, k])
#     data.append(row)

# if SPLIT > len(data):
#     raise ValueError

# file = open("data.txt", "w")
# for row in data:
#     file.write(f'{str(row)}\n')
# file.close()
# file = open("targets.txt", "w")
# for row in targets:
#     file.write(f'{str(row)}\n')
# file.close()

data = []
targets = []

file = open("data.txt", "r")
for line in file:
    line = line.strip(" \n[]")
    line = line.split(", ")
    data_line = [element.strip("'") for element in line]
    data.append(data_line)
file.close()
file = open("targets.txt", "r")
for element in file:
    element = element.strip(" \n")
    targets.append(element)
file.close()

# [:2] - do 2 elementu wyłącznie, [2:] od 2 elementu włącznie
training_data = data[:SPLIT]
testing_data = data[SPLIT:]
training_targets = targets[:SPLIT]
testing_targets = targets[SPLIT:]

forest = create_forest(training_data, training_targets, TREES)

results = []
for index, row in enumerate(testing_data):
    print(index)
    results.append((testing_targets[index], get_forest_classifier(forest, row)))

dictionary = get_classification_dict(results)
print(round(get_accuracy(dictionary, CLASSES), 2))
print(round(get_recall(dictionary, CLASSES), 2))
print(round(get_precision(dictionary, CLASSES), 2))
print("finished")
