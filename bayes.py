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

        part1 = (1 / np.sqrt(2 * np.pi * np.square(dev)))
        part2 = np.exp(-((np.square(feature_value - mean)) / (2 * np.square(dev))))
        return part1 * part2

    def train(self):
        self.calculate_priors()

        predictions = []
        for row in self.data:
            probabilities = [1] * len(self.class_priors)
            for prior_index, prior in enumerate(self.class_priors):
                for feature_index, feature in enumerate(row):
                    probabilities[prior_index] *= self.calculate_prob(feature_index, feature, prior)

            post_probabilities = [1] * len(self.class_priors)
            for prior_index, prior in enumerate(self.class_priors):
                post_probabilities[prior_index] = probabilities[prior_index] * self.class_priors[prior]

            predictions.append(np.argmax(post_probabilities))

        return predictions


    def test(self, features):
        probabilities = [1] * len(self.class_priors)
        for prior_index, prior in enumerate(self.class_priors):
            for feature_index, feature in enumerate(features):
                probabilities[prior_index] *= self.calculate_prob(feature_index, feature, prior)

        post_probabilities = [1] * len(self.class_priors)
        for prior_index, prior in enumerate(self.class_priors):
            post_probabilities[prior_index] = probabilities[prior_index] * self.class_priors[prior]

        prediction = np.argmax(post_probabilities)

        return prediction







