from random import random
import numpy as np

class NeuralNet:
    def __init__(self, data, targets, learning_speed):
        self.data = data
        self.targets = targets
        self.learning_speed = learning_speed
        self.layer_weights = {}
        self.layer_biases = {}

    def activation_function(value):
        return 1 / (1 + np.exp(-value))

    def create_net(self, hidden_layers, hidden_neurons):
        self.layer_weights[0] = []
        self.layer_biases[0] = random()
        for _ in range(len(self.data[0])):
            self.layer_weights[0].append(random())
        
        for i in range(1, hidden_layers + 1):
            self.layer_weights[i] = []
            self.layer_biases[i] = random()
            for j in range(hidden_neurons):
                self.layer_weights[i].append(random())
        
        self.layer_weights[hidden_layers + 1] = []
        self.layer_biases[hidden_layers + 1] = random()
        for _ in range(len(np.unique(self.targets))):
            self.layer_weights[hidden_layers + 1].append(random())


    def train(self, cycles):
        pass
    