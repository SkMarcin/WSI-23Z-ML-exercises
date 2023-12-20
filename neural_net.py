from neurons import InputNeuron, DeepNeuron, OutputNeuron
from random import random, choice, randint
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

    def create_net(self, deep_layer_count, deep_neuron_count):
        """
        This method creates the structure of NeuralNet
        
        input_layer contains InputNeurons with only a value
        
        deep_layers contains deep_layer_count layers with deep_neuron_count DeepNeurons each.
        
        output_layer contains as many neurons as there are unique classes in targets
        """
        for _ in range(len(self.data[0])):
            self.input_layer.append(InputNeuron())
        
        for i in range(deep_layer_count):
            self.deep_layers.append([])
            if i == 0:
                for _ in range(deep_neuron_count):
                    weights = [random() for _ in range(len(self.input_layer))]
                    bias = random()
                    self.deep_layers[i].append(DeepNeuron(weights, bias))
            else:
                for _ in range(deep_neuron_count):
                    weights = [random() for _ in range(deep_neuron_count)]
                    bias = random()
                    self.deep_layers[i].append(DeepNeuron(weights, bias))
        
        if deep_layer_count == 0:
            # when there are no deep layers
            deep_neuron_count = len(self.input_layer)
    
        for _ in range(len(np.unique(self.targets))):
            weights = [random() for _ in range(deep_neuron_count)]
            bias = random()
            self.output_layer.append(OutputNeuron(weights, bias))

    def train_cycle(self):
        # forward propagation part
        index = randint(0, len(self.data))
        self.load_pixels(index)

        prev_layer_values = []
        this_layer_values = []
        for neuron in self.input_layer:
            prev_layer_values.append(neuron.value)

        for i, layer in enumerate(self.deep_layers):
            for neuron in layer:
                neuron.value = np.dot(neuron.weights, prev_layer_values) + neuron.bias
                neuron.value = self.activation_function(neuron.value)
                this_layer_values.append(neuron.value)
            prev_layer_values = this_layer_values
            this_layer_values = []

        for neuron in self.output_layer:
            neuron.value = np.dot(neuron.weights, prev_layer_values) + neuron.bias
            neuron.value = self.activation_function(neuron.value)
            this_layer_values.append(neuron.value)

        #back propagation part



    def load_pixels(self, index):
        """this method inputs pixel data from index into input layer"""

        for pixel in self.data[index]:
            self.input_layer[index].value = pixel
    