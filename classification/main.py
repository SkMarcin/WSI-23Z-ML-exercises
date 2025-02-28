from ucimlrepo import fetch_ucirepo
from training import create_forest, get_forest_classifier, get_classification_dict, get_accuracy, get_recall, get_precision, get_fallout, get_f1score
import numpy as np
import matplotlib.pyplot as plt

SPLIT = 200
TREES = 200
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

for index, row in enumerate(data):
    row.append(targets[index])

np.random.shuffle(data)

targets = [row[6] for row in data]

for row in data:
    row.pop()

# [:2] - do 2 elementu wyłącznie, [2:] od 2 elementu włącznie
training_data = data[SPLIT:]
testing_data = data[:SPLIT]
training_targets = targets[SPLIT:]
testing_targets = targets[:SPLIT]

forest = create_forest(training_data, training_targets, TREES)

results = []
for index, row in enumerate(testing_data):
    results.append((testing_targets[index], get_forest_classifier(forest, row)))

dictionary = get_classification_dict(results)
# print(round(get_accuracy(dictionary, CLASSES), 2))
# print(round(get_recall(dictionary, CLASSES), 2))
# print(round(get_precision(dictionary, CLASSES), 2))
# print(round(get_fallout(dictionary, CLASSES), 2))
# print(round(get_f1score(dictionary, CLASSES), 2))


accuracies = []
recalls = []
precisions = []
fallouts = []
f1scores = []
for i in range(10):
    forest = create_forest(training_data, training_targets, TREES)

    results = []
    for index, row in enumerate(testing_data):
        results.append((testing_targets[index], get_forest_classifier(forest, row)))

    dictionary = get_classification_dict(results)

    accuracies.append(get_accuracy(dictionary, CLASSES))
    recalls.append(get_recall(dictionary, CLASSES))
    precisions.append(get_precision(dictionary, CLASSES))
    fallouts.append(get_fallout(dictionary, CLASSES))
    f1scores.append(get_f1score(dictionary, CLASSES))

print(round(np.mean(accuracies), 2))
print(round(np.mean(recalls), 2))
print(round(np.mean(precisions), 2))
print(round(np.mean(fallouts), 2))
print(round(np.mean(f1scores), 2))

print("odchylenia")
print(np.std(accuracies, ddof=1))
print(np.std(recalls, ddof=1))
print(np.std(precisions, ddof=1))
print(np.std(fallouts, ddof=1))
print(np.std(f1scores, ddof=1))
