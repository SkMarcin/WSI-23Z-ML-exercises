from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from neural_net import NeuralNet
import numpy as np

LEARNING_SPEED = 0.1

mnist = fetch_openml('mnist_784', version=1)

data = np.array(mnist.data)
targets = np.array(mnist.target)


training_data, temp_data, training_targets, temp_targets = train_test_split(data, targets, test_size=0.2, random_state=1)
testing_data, validation_data, testing_targets, validation_targets = train_test_split(temp_data, temp_targets, test_size=0.5, random_state=1)

network = NeuralNet(training_data, training_targets, LEARNING_SPEED)
network.create_net(5, 15)
