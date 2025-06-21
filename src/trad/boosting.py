import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

loaded = load_breast_cancer()
X = loaded.data
y = loaded.target

y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)

class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = None
        self.alpha = None

    def predict(self, X):
        n = X.shape[0]
        preds = np.ones(n)
        if self.polarity == 1:
            preds[X[:, self.feature_index] < self.threshold] = -1
        else:
            preds[X[:, self.feature_index] > self.threshold] = -1
        return preds


import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = data['data']
y = data['target']

# Convert y to {-1, 1}
y = np.where(y == 0, -1, 1)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision stump (1-feature threshold classifier)
class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def predict(self, X):
        n = X.shape[0]
        preds = np.ones(n)
        if self.polarity == 1:
            preds[X[:, self.feature_index] < self.threshold] = -1
        else:
            preds[X[:, self.feature_index] > self.threshold] = -1
        return preds

class AdaBoostCustom:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.stumps = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            min_error = float('inf')

            for feature_i in range(n_features):
                thresholds = np.unique(X[:, feature_i])
                for thresh in thresholds:
                    for polarity in [1, -1]:
                        preds = np.ones(n_samples)
                        if polarity == 1:
                            preds[X[:, feature_i] < thresh] = -1
                        else:
                            preds[X[:, feature_i] > thresh] = -1

                        error = np.sum(w * (preds != y))

                        if error < min_error:
                            min_error = error
                            stump.feature_index = feature_i
                            stump.threshold = thresh
                            stump.polarity = polarity

            epsilon = 1e-10  # smoothing
            stump.alpha = 0.5 * np.log((1 - min_error + epsilon) / (min_error + epsilon))

            preds = stump.predict(X)
            w *= np.exp(-stump.alpha * y * preds)
            w /= np.sum(w)

            self.stumps.append(stump)

    def predict(self, X):
        stump_preds = [stump.alpha * stump.predict(X) for stump in self.stumps]
        final_pred = np.sign(np.sum(stump_preds, axis=0))
        return final_pred

model = AdaBoostCustom(n_estimators=20)
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print(f"Train: {accuracy_score(y_train, train_pred):.4f}")
print(f"Test:  {accuracy_score(y_test, test_pred):.4f}")

