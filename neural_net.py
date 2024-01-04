from neurons import InputNeuron, DeepNeuron, OutputNeuron
from random import random, choice, randint
import numpy as np
import matplotlib.pyplot as plt
from time import time

class NeuralNet:
    def __init__(self, sizes=[784, 64, 10], repeat=10, learing_speed=0.1):
        self.sizes = sizes
        self.repeat = repeat
        self.learning_speed = learing_speed
        self.weights = [[]]
        self.pre_activation = [[]]
        self.post_activation = [[]]

        input_layer = sizes[0]
        deepSizes = []
        for i in range(1, len(sizes) - 1):
            deepSizes.append(sizes[i])
        output_layer = sizes[-1]
        print(input_layer)
        print(deepSizes)
        print(output_layer)

        for i in range(len(sizes) - 1):
            j = i + 1
            self.weights.append([])
            self.weights[j] = np.random.randn(sizes[j], sizes[i]) * np.sqrt(1./sizes[j])
            self.pre_activation.append([])
            self.post_activation.append([])

        # print(self.params)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self, x):
        return (np.exp(-x))/((np.exp(-x)+1)**2)

    def softmax(self, x):
        exps = np.exp(x-x.max())
        return exps / np.sum(exps, axis=0)

    def softmax_derivative(self, x):
        exps = np.exp(x-x.max())
        return exps / np.sum(exps, axis=0) * (1-exps / np.sum(exps, axis=0))

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
        error = 2 * (output - y_train) / output.shape[0] * self.softmax_derivative(self.pre_activation[last_id])
        change_w[last_id] = np.outer(error, self.post_activation[last_id-1])

        a = last_id - 2
        while a >= 0:
            error = np.dot(self.weights[a+2].T, error) * self.sigmoid_derivative(self.pre_activation[a+1])
            change_w[a+1] = np.outer(error, self.post_activation[a])
            a -= 1

        return change_w

    def update_weights(self, change_w):
        for key, val in change_w.items():
            self.weights[key] -= val * self.learning_speed

    def get_accuracy(self, train_list):
        predictions = []
        classes_dict = {}
        classes_correct = {}
        for x in train_list:
            values = x
            inputs = (np.asfarray(values[1:])/255*0.99) + 0.01
            targets = np.zeros(10) + 0.01
            targets[int(values[0])] = 0.99
            output = self.foward_pass(inputs)
            prediction = np.argmax(output)
            predictions.append(prediction==np.argmax(targets))
            if prediction in classes_dict:
                classes_dict[prediction] += 1
            else:
                classes_dict[prediction] = 1
            if prediction == np.argmax(targets):
                if prediction in classes_correct:
                    classes_correct[prediction] += 1
                else:
                    classes_correct[prediction] = 1

        # print(classes_dict)
        # print(classes_correct)
        return np.mean(predictions)


    def train(self, train_list, test_list):
        val_accuracies = []
        train_accuracies = []
        iterations = []
        start = time()
        counter = 0
        for i in range(self.repeat):
            a = 0
            for index, x in enumerate(train_list):
                values = x
                inputs = (np.asfarray(values[1:])/255*0.99) + 0.01
                targets = np.zeros(10) + 0.01
                targets[int(values[0])] = 0.99
                output = self.foward_pass(inputs)
                change_w = self.backward_pass(targets, output)
                self.update_weights(change_w)
                # print(f"Looping {a}")
                a += 1
                counter += 1
                if counter % 5000 == 0:
                    accuracy = self.get_accuracy(test_list)
                    train_accuracy = self.get_accuracy(train_list)
                    val_accuracies.append(accuracy)
                    train_accuracies.append(train_accuracy)
                    iterations.append(counter)
                    print(f"Repeated: {accuracy}")


            print(f"{time() - start}")

        plt.plot(iterations, val_accuracies, label="validation")
        plt.plot(iterations, train_accuracies, label="training")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.title("Accuracy graph")
        plt.legend()
        plt.show()
