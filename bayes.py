import numpy as np

class BayesClassifier:
    def __init__(self, ):
        pass

    def fit(self, data, targets):
        self.targets = np.unique(targets)
        self.class_priors = {}
        self.class_means = {}
        self.class_variances = {}

        for target in self.targets:
            for row in data:
                self.class_priors[target] += 1
                self.class_means[target] = np.mean(X_c, axis=0)
                self.class_variances[target] = np.var(X_c, axis=0)