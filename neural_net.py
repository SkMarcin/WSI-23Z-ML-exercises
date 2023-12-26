from neurons import InputNeuron, DeepNeuron, OutputNeuron
from random import random, choice, randint
import numpy as np

class NeuralNet:
    def __init__(self, sizes=[784, 128, 64, 10], repeat=10, learing_speed=0.001):
        self.sizes = sizes
        self.repeat = repeat
        self.learning_speed = learing_speed

        input_layer = sizes[0]
        deepSizes = []
        for i in range(1, len(sizes) - 1):
            deepSizes.append(sizes[i])
        output_layer = sizes[-1]
        print(deepSizes)
        print(output_layer)

        self.params = {}

        for i in range(len(sizes) - 1):
            j = i + 1
            key = "W" + str(j)
            self.params[key] = np.random.randn(sizes[j], sizes[i]) * np.sqrt(1./sizes[j])

        self.initial_params = self.params
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
        params = self.params

        params['A0'] = x_train
        last_id = 0
        for i in range(len(self.params) - 1):
            j = i + 1
            params['Z'+str(j)] = np.dot(params['W'+str(j)], params['A'+str(i)])
            params['A'+str(j)] = self.sigmoid(params['Z'+str(j)])
            last_id = j
        print(last_id)
        params['Z'+str(last_id)] = np.dot(params['W'+str(last_id)], params['A'+str(last_id-1)])
        params['A'+str(last_id)] = self.softmax(params['Z'+str(last_id)])

        return params['Z'+str(last_id)]

    def get_last_number(s):
        i = len(s) - 1

        while i >= 0 and s[i].isdigit():
            i -= 1

        last_number = s[i + 1:] if i < len(s) - 1 else None
        return int(last_number) if last_number is not None else None


    def backward_pass(self, y_train, output):
        params = self.params

        change_w = {}
        last_key = list(params.keys())[-1]
        last_id = self.get_last_number(last_key)
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params['Z'+str(last_id)], True)
        change_w['W'+str(last_id)] = np.outer(error, params['A'+str(last_id-1)])

        a = last_id - 2
        while a >= 0:
            error = np.dot(params['W' + str(a+2)].T, error) * self.sigmoid(params['Z'+str(a+1)], True)
            change_w['W'+str(a+1)] = np.outer(error, params['A'+str(a)])

        return change_w

    def update_weights(self, change_w):
        for key, val in change_w.items():
            self.params[key] -= val * self.learning_speed

    def get_accuracy(self, train_list):
        predictions = []
        for x in train_list:
            values = train_list.to_list()
            inputs = (np.asfarray(values[1:])/255*0.99) + 0.01
            targets = np.zeros(10) + 0.01
            targets[int(values[0])] = 0.99
            output = self.foward_pass(inputs)
            prediction = np.argmax(output)
            predictions.append(prediction==np.argmax(targets))
        return np.mean(predictions)


    def train(self, train_list, test_list):
        for i in range(self.repeat):
            for x in train_list:
                values = train_list.to_list()
                inputs = (np.asfarray(values[1:])/255*0.99) + 0.01
                targets = np.zeros(10) + 0.01
                targets[int(values[0])] = 0.99
                output = self.foward_pass(inputs)
                change_w = self.backward_pass(targets, output)
                self.update_weights(change_w)

            accuracy = self.get_accuracy(test_list)
