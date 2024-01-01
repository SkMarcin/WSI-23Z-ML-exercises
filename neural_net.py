from neurons import InputNeuron, DeepNeuron, OutputNeuron
from random import random, choice, randint
import numpy as np

class NeuralNet:
    def __init__(self, sizes=[784, 64, 64, 10], repeat=10, learing_speed=0.1):
        self.sizes = sizes
        self.repeat = repeat
        self.learning_speed = learing_speed
        self.weights = [0]
        self.pre_activation = [[]]
        self.post_activation = [[]]

        input_layer = sizes[0]
        deepSizes = []
        for i in range(1, len(sizes) - 1):
            deepSizes.append(sizes[i])
        output_layer = sizes[-1]
        print(deepSizes)
        print(output_layer)

        for i in range(len(sizes) - 1):
            j = i + 1
            self.weights.append([])
            self.weights[j] = np.random.randn(sizes[j], sizes[i]) * np.sqrt(1./sizes[j])
            self.pre_activation.append([])
            self.post_activation.append([])

        # print(self.params)

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1+np.exp(-x))

    def softmax(self, x, derivative=False):
        exps = np.exp(x-x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1-exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def foward_pass(self, x_train):
        last_id = self.get_last_number()
        i = 0

        self.post_activation[i] = x_train

        while i < last_id:
            j = i + 1
            self.pre_activation[j] = np.dot(self.weights[j], self.post_activation[i])
            self.post_activation[j] = self.sigmoid(self.pre_activation[j])
            i += 1

        # print(last_id)
        self.pre_activation[last_id] = np.dot(self.weights[last_id], self.post_activation[last_id-1])
        self.post_activation[last_id] = self.softmax(self.pre_activation[last_id])

        return self.post_activation[last_id]

    def get_last_number(self):
        return len(self.weights) - 1


    def backward_pass(self, y_train, output):
        change_w = {}
        last_id = self.get_last_number()
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(self.pre_activation[last_id], True)
        change_w[last_id] = np.outer(error, self.post_activation[last_id-1])

        a = last_id - 2
        while a >= 0:
            error = np.dot(self.weights[a+2].T, error) * self.sigmoid(self.pre_activation[a+1], True)
            change_w[a+1] = np.outer(error, self.post_activation[a])
            a -= 1

        return change_w

    def update_weights(self, change_w):
        for key, val in change_w.items():
            self.weights[key] -= val * self.learning_speed

    def get_accuracy(self, train_list):
        predictions = []
        for x in train_list:
            values = x
            inputs = (np.asfarray(values[1:])/255*0.99) + 0.01
            targets = np.zeros(10) + 0.01
            targets[int(values[0])] = 0.99
            output = self.foward_pass(inputs)
            prediction = np.argmax(output)
            predictions.append(prediction==np.argmax(targets))
        return np.mean(predictions)


    def train(self, train_list, test_list):
        for i in range(self.repeat):
            a = 0
            for x in train_list:
                values = x
                inputs = (np.asfarray(values[1:])/255*0.99) + 0.01
                targets = np.zeros(10) + 0.01
                targets[int(values[0])] = 0.99
                output = self.foward_pass(inputs)
                change_w = self.backward_pass(targets, output)
                self.update_weights(change_w)
                # print(f"Looping {a}")
                a += 1

            accuracy = self.get_accuracy(test_list)
            print(f"Repeated: {accuracy}")
