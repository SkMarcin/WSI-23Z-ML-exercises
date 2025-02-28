import numpy as np

class BayesClassifier:
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def calculate_priors(self):
        targets = np.unique(self.targets)
        self.class_priors = {}

        for target in targets:
            for index, row in enumerate(self.data):
                if self.targets[index] == target:
                    if target in self.class_priors:
                        self.class_priors[target] += 1
                    else:
                        self.class_priors[target] = 1

            self.class_priors[target] /= len(self.targets)

    def calculate_prob(self, feature_index, feature_value, target):
        feature_values = []
        for index, row in enumerate(self.data):
            if self.targets[index] == target:
                feature_values.append(row[feature_index])

        mean = np.mean(feature_values)
        dev = np.std(feature_values)

        return (1 / np.sqrt(2 * np.pi * np.square(dev))) * np.exp(-((np.square(feature_value - mean)) / (2 * np.square(dev))))

    def train(self):
        self.calculate_priors()




    def test(self, features):
        probabilities = [1] * len(self.class_priors)
        for prior_index, prior in enumerate(self.class_priors):
            for feature_index, feature in enumerate(features):
                probabilities[prior_index] *= self.calculate_prob(feature_index, feature, prior)

        post_probabilities = [1] * len(self.class_priors)
        for prior_index, prior in enumerate(self.class_priors):
            post_probabilities[prior_index] = probabilities[prior_index] * self.class_priors[prior]

        prediction = np.argmax(post_probabilities)

        return prediction + 1



def get_classification_dict(tuples):
    dict = {('1', '1'): 0, ('1', '2'): 0, ('1', '3'): 0,
            ('2', '1'): 0, ('2', '2'): 0, ('2', '3'): 0,
            ('3', '1'): 0, ('3', '2'): 0, ('3', '3'): 0,
    }
    for tuple in tuples:
        dict[tuple] += 1
    return dict

def get_accuracy(dictionary, classes):
    correct = 0
    incorrect = 0
    for classifier in classes:
        for key in dictionary:
            if key[0] == key[1]:
                correct += dictionary[key]
            else:
                incorrect += dictionary[key]
    return correct / (correct + incorrect)

def get_precision(dictionary, classes):
    precisions = []
    for classifier in classes:
        correct = 0
        total = 0
        for key in dictionary:
            if key[1] == classifier:
                if key[0] == classifier:
                    correct += dictionary[key]
                total += dictionary[key]
        if total > 0:
            precisions.append(correct/total)
    return sum(precisions) / len(precisions)

def get_sensitivity(dictionary, classes):
    precisions = []
    for classifier in classes:
        true_positives = 0
        total = 0
        for key in dictionary:
            if key[0] == classifier:
                if key[1] == classifier:
                    true_positives += dictionary[key]
                total += dictionary[key]
        if total > 0:
            precisions.append(true_positives/total)
    return sum(precisions) / len(precisions)