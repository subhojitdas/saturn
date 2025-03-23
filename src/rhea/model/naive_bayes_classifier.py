import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class NaiveBayes:

    def __init__(self):
        self.class_probs = {}
        self.feature_probs = {}

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        for cls in self.classes:
            X_cls = X[y == cls]
            self.class_probs[cls] = len(X_cls) / n_samples

            self.feature_probs[cls] = {
                "mean": X_cls.mean(axis=0),
                "var": X_cls.var(axis=0) + 1e-6
            }

    def _gaussian_likelihood(self, cls, x):
        mean = self.feature_probs[cls]["mean"]
        var = self.feature_probs[cls]["var"]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict(self, X):
        predictions = []
        for x in X:
            class_probs = {}
            for cls in self.classes:
                prior = np.log(self.class_probs[cls])
                # MLE
                likelihood = np.sum(np.log(self._gaussian_likelihood(cls, x)))
                class_probs[cls] = prior + likelihood
            predictions.append(max(class_probs, key=class_probs.get))
        return np.array(predictions)

np.random.seed(0)
X = np.vstack((np.random.normal(0, 1, (100, 2)), np.random.normal(2, 1, (100, 2))))
y = np.array([0] * 100 + [1] * 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = NaiveBayes()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
