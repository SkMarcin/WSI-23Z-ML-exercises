from bayes import BayesClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns

file_handle = open("seeds_dataset.txt", "r")

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

training_data, testing_data, training_targets, testing_targets = train_test_split(data, targets, test_size=0.2)


bayes = BayesClassifier(training_data, training_targets)
bayes.train()
results = []

for index, line in enumerate(testing_data):
    string = bayes.test(line)
    if int(bayes.test(line)) == int(testing_targets[index]) - 1:
        results.append(True)
    else:
        results.append(False)

print('fin')