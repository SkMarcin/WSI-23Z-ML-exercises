from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from neural_net import NeuralNet
import matplotlib.pyplot as plt
import numpy as np

LEARNING_SPEED = 0.001
HOW_MANY_CYCLES = 6000
CHECK_FREQUENCY = 10
DOT_SIZE = 0.2

def plot_errors(errors, indexes):
    plt.scatter(indexes, errors, marker='o', s=DOT_SIZE)
    plt.title("Error changing")
    plt.xlabel("Training ammount")
    plt.ylabel("Error value")
    plt.show()

def main():
    mnist = fetch_openml('mnist_784', version=1)

    print("downloaded")

    data = np.array(mnist.data)
    targets = np.array(mnist.target)

    training_data, testing_data, training_targets, testing_targets = train_test_split(data, targets, test_size=0.2, random_state=1)

    training_array = []
    temp_list = training_data.tolist()
    for i in range(len(temp_list)):
        target = [int(training_targets[i])]
        img = temp_list[i]
        temp = target + img
        training_array.append(temp)

    testing_array = []
    temp_list = testing_data.tolist()
    for i in range(len(temp_list)):
        target = [int(testing_targets[i])]
        img = temp_list[i]
        temp = target + img
        testing_array.append(temp)

    print("devided")
    network = NeuralNet()
    print("Initialized")
    network.train(training_array, testing_array)


if __name__ == "__main__":
    main()