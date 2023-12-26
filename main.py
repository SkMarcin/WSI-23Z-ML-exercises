from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from neural_net import NeuralNet
import matplotlib.pyplot as plt
import numpy as np

LEARNING_SPEED = 0.001
HOW_MANY_CYCLES = 60000
CHECK_FREQUENCY = 10
DOT_SIZE = 0.2

def simulate_cycles(network):
    errors = []
    indexes = []
    for i in range(HOW_MANY_CYCLES):
        network.train_cycle()
        if i % CHECK_FREQUENCY == 0:
            indexes.append(i)
            errors.append(network.get_output_error())
    return errors, indexes

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

    np.random.shuffle(data)

    training_data, temp_data, training_targets, temp_targets = train_test_split(data, targets, test_size=0.2, random_state=1)
    testing_data, validation_data, testing_targets, validation_targets = train_test_split(temp_data, temp_targets, test_size=0.5, random_state=1)
    print("devided")
    network = NeuralNet()
    print("Initialized")
    network.train(training_data, testing_data)


if __name__ == "__main__":
    main()