from bayes import BayesClassifier, get_classification_dict, get_accuracy, get_precision, get_sensitivity
from sklearn.model_selection import train_test_split
# import pandas as pd
# import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

CLASSES =['1', '2', '3']

file_handle = open("./seeds_dataset.txt", "r")



data = []
targets = []

# for line in file_handle:
#     line = line.strip(" \n[]")
#     line = line.split("\t")
#     line = [element for element in line if element]
#     data_line = []
#     for i in range(7):
#         data_line.append(float(line[i]))

#     data_line.append(line[7])
#     data.append(data_line)
#     targets.append(line[7])


# df = pd.DataFrame(data, columns=['area', 'perimeter', 'compactness', 'length', 'width', 'assymetry_coeff', 'groove_length', 'class'])
# corr = df.iloc[:,:-1].corr(method="pearson")
# cmap = sns.diverging_palette(250,354,80,60,center='dark',as_cmap=True)
# sns.heatmap(corr, vmax=1, vmin=-0.5, cmap=cmap, square=True, linewidths=.2)


for line in file_handle:
    line = line.strip(" \n[]")
    line = line.split("\t")
    line = [element for element in line if element]
    data_line = []
    data_line.append(float(line[2]))
    data_line.append(float(line[5]))
    data_line.append(float(line[6]))
    data.append(data_line)
    targets.append(line[7])


# index = 2
# targets = np.array(targets)
# values = np.array([line[index] for line in data])

# class_0_values = values[targets == '1']
# class_1_values = values[targets == '2']
# class_2_values = values[targets == '3']

# hist_bins = np.linspace(min(values), max(values), 30)
# hist_class_0, _ = np.histogram(class_0_values, bins=hist_bins)
# hist_class_1, _ = np.histogram(class_1_values, bins=hist_bins)
# hist_class_2, _ = np.histogram(class_2_values, bins=hist_bins)

# bar_width = 0.8 * (hist_bins[1] - hist_bins[0])
# plt.bar(hist_bins[:-1], hist_class_0, width=bar_width, label='Kama', alpha=0.7)
# plt.bar(hist_bins[:-1], hist_class_1, width=bar_width, label='Rosa', alpha=0.7, bottom=hist_class_0)
# plt.bar(hist_bins[:-1], hist_class_2, width=bar_width, label='Canadian', alpha=0.7, bottom=hist_class_0 + hist_class_1)

# plt.xlabel('Values')
# plt.ylabel('Count')
# plt.title('Histogram of Kernel Grove Length Values for Each Class')
# plt.legend()
# plt.grid(True)
# plt.show()

# train_size = 0.05
# training_data = data[:int(len(data)*train_size)]
# training_targets = targets[:int(len(data)*train_size)]

# testing_data = data[int(len(data)*train_size):]
# testing_targets = targets[int(len(data)*train_size):]

# results_total = []
# for i in range(100):
training_data, testing_data, training_targets, testing_targets = train_test_split(data, targets, test_size=0.5)

bayes = BayesClassifier(training_data, training_targets)
bayes.train()
results = []

predicted = []

for index, line in enumerate(testing_data):
    predicted.append(str(bayes.test(line)))
    if int(bayes.test(line)) == int(testing_targets[index]):
        results.append(True)
    else:
        results.append(False)

dict = get_classification_dict(list(zip(testing_targets, predicted)))

print(get_accuracy(dict, CLASSES))
print(get_precision(dict, CLASSES))
print(get_sensitivity(dict, CLASSES))

print(dict)
print('fin')