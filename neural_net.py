from neurons import InputNeuron, DeepNeuron, OutputNeuron
from random import random
import numpy as np

class NeuralNet:
    def __init__(self, data, targets, learning_speed):
        self.data = data
        self.targets = targets
        self.learning_speed = learning_speed
        self.input_layer = []
        self.deep_layers = []
        self.output_layer = []

    def activation_function(value):
        return 1 / (1 + np.exp(-value))

    def create_net(self, deep_layer_count, deep_neurons):
        for _ in range(len(self.data[0])):
            self.input_layer.append(InputNeuron())
        
        for i in range(deep_layer_count):
            self.deep_layers.append([])
            if i == 0:
                for _ in range(deep_neurons):
                    weights = [random() for _ in range(len(self.input_layer))]
                    bias = random()
                    self.deep_layers[i].append(DeepNeuron(weights, bias))
            else:
                for _ in range(deep_neurons):
                    weights = [random() for _ in range(deep_neurons)]
                    bias = random()
                    self.deep_layers[i].append(DeepNeuron(weights, bias))
        
        for _ in range(len(np.unique(self.targets))):
            weights = [random() for _ in range(deep_neurons)]
            bias = random()
            self.output_layer.append(OutputNeuron(weights, bias))

    def train(self, cycles):
        pass
    