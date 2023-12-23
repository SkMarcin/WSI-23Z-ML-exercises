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

    def activation_function(self, value):
        return 1 / (1 + np.exp(-value))

    def activation_derivative(self, value):
        return self.activation_function(value) * (1 - self.activation_function(value))

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
        """
        This method runs a complete training cycle for the neural net including
        forward, backward propagation as well as updating weights and biases
        """

        # forward propagation part
        index = randint(0, len(self.data) - 1)
        self.load_pixels(index)
        target_class = self.targets[index]

        prev_layer_values = []
        this_layer_values = []
        for neuron in self.input_layer:
            prev_layer_values.append(neuron.value)

        for layer in self.deep_layers:
            for neuron in layer:
                neuron.value = np.dot(neuron.weights, prev_layer_values) + neuron.bias
                neuron.value = self.activation_function(neuron.value)
                neuron.derivative_value = self.activation_derivative(neuron.value)  # this is used
                neuron.weights_derivatives = np.array(prev_layer_values)            # in back propagation
                this_layer_values.append(neuron.value)
            prev_layer_values = this_layer_values
            this_layer_values = []

        for neuron in self.output_layer:
            neuron.value = np.dot(neuron.weights, prev_layer_values) + neuron.bias
            neuron.value = self.activation_function(neuron.value)
            neuron.derivative_value = self.activation_derivative(neuron.value)      # this is used
            neuron.weights_derivatives = np.array(prev_layer_values)                # in back propagation
            this_layer_values.append(neuron.value)

        # calculating error
        probabilities = self.softmax()
        errors = self.cross_entropy(probabilities, target_class)

        for index, error in enumerate(errors):
            self.output_layer[index].error = error

        # back propagation part
        next_layer_weights = []
        next_layer_errors = []
        for neuron in self.output_layer:
            next_layer_weights.append(neuron.weights)
            next_layer_errors.append(neuron.error)

        # update output_layer parameters
        for neuron in self.output_layer:
            weights_gradients = neuron.weights_derivatives * neuron.derivative_value * neuron.error
            bias_gradient = neuron.derivative_value * neuron.error
            neuron.weights = neuron.weights - self.learning_speed * weights_gradients
            neuron.bias = neuron.bias - self.learning_speed * bias_gradient

        for layer in reversed(self.deep_layers):
            # propagate error
            for neuron_index, neuron in enumerate(layer):
                neuron.error = 0
                for error_index, error in enumerate(next_layer_errors):
                    neuron.error += error * next_layer_weights[error_index][neuron_index]

            next_layer_weights = []
            next_layer_errors = []
            for neuron in layer:
                next_layer_weights.append(neuron.weights)
                next_layer_errors.append(neuron.error)

            # update layer parameters
            for neuron in layer:
                weights_gradients = neuron.weights_derivatives * neuron.derivative_value * neuron.error
                bias_gradient = neuron.derivative_value * neuron.error
                neuron.weights = neuron.weights - self.learning_speed * weights_gradients
                neuron.bias = neuron.bias - self.learning_speed * bias_gradient


    def load_pixels(self, index):
        """this method inputs pixel data from index into input layer"""

        for pixel_index, pixel in enumerate(self.data[index]):
            self.input_layer[pixel_index].value = pixel

    def softmax(self):
        """this method returns the probabilities for each class according to the network"""
        logits_sum = 0
        probabilities = []
        for neuron in self.output_layer:
            logits_sum += np.exp(neuron.value)

        for neuron in self.output_layer:
            probabilities.append(np.exp(neuron.value) / logits_sum)

        return probabilities

    def cross_entropy(self, probabilities, target_class):
        """this method calculates error for class probabilities"""
        epsilon = 1e-12

        errors = []
        for index, probabilty in enumerate(probabilities):
            probabilty = np.clip(probabilty, epsilon, 1)
            if index == int(target_class):
                errors.append(-probabilty * np.log(probabilty))
            else:
                errors.append(0)

        return errors

    def get_output_error(self):
        total = 0
        for outputNeuron in self.output_layer:
            total += outputNeuron.error
        return total

    def make_prediction(self, pixels, classed):
        # load input layer
        for i in range(len(pixels)):
            self.input_layer[i].value = pixels[i]
        target_class = classed

        prev_layer_values = []
        this_layer_values = []
        for neuron in self.input_layer:
            prev_layer_values.append(neuron.value)

        for layer in self.deep_layers:
            for neuron in layer:
                neuron.value = np.dot(neuron.weights, prev_layer_values) + neuron.bias
                neuron.value = self.activation_function(neuron.value)
                neuron.derivative_value = self.activation_derivative(neuron.value)  # this is used
                neuron.weights_derivatives = np.array(prev_layer_values)            # in back propagation
                this_layer_values.append(neuron.value)
            prev_layer_values = this_layer_values
            this_layer_values = []

        for neuron in self.output_layer:
            neuron.value = np.dot(neuron.weights, prev_layer_values) + neuron.bias
            neuron.value = self.activation_function(neuron.value)
            neuron.derivative_value = self.activation_derivative(neuron.value)      # this is used
            neuron.weights_derivatives = np.array(prev_layer_values)                # in back propagation
            this_layer_values.append(neuron.value)

        # calculating error
        probabilities = self.softmax()
        calculated_class = probabilities.index(max(probabilities))
        correct = calculated_class == int(target_class)

        return calculated_class, correct

    def validate_network(self, dataset, classes):
        successes = 0
        for ind in range(len(dataset)):
            img = dataset[ind]
            classed = classes[ind]
            _, success = self.make_prediction(img, classed)
            if success:
                successes += 1
        return successes / len(dataset)