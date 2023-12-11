import matplotlib.pyplot as plt
from main import CLASSES
import numpy as np



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


val = (np.arange(4))
width = 0.15

values_for_graph = []
for classifier in CLASSES:
    attributes_dict = {"low": 0, "med": 0, "high": 0}
    for index, row in enumerate(data):
        if targets[index] == classifier:
            attributes_dict[row[5]] += 1
    array = []
    for key in attributes_dict:
        array.append(attributes_dict[key])
    values_for_graph.append(array)

converted_array = np.transpose(values_for_graph)

plt.clf()
values = converted_array[0]
bar0 = plt.bar(val, values, width, color = 'r')

values = converted_array[1]
bar1 = plt.bar(val+width, values, width, color = 'g')

values = converted_array[2]
bar2 = plt.bar(val+2*width, values, width, color = 'b')

# values = converted_array[3]
# bar3 = plt.bar(val+3*width, values, width, color = 'y')



plt.xlabel("Classes")
plt.ylabel('Amount in class')
plt.title("Safety per class")

plt.xticks(val+width, CLASSES)
plt.legend( (bar0, bar1, bar2), ("low", "med", "high") )
plt.show()

