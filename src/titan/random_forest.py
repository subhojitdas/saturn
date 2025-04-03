import numpy as np
from collections import Counter


from src.titan.decision_tree import DecisionTreeClassifier
from src.titan.fast_decision_tree import FastDecisionTreeClassifier


class RandomForestClassifierSub:
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def bootstrap_sample(self, X, Y):
        n_samples, num_features = X.shape
        sample_indices = np.random.choice(n_samples, n_samples, replace=True)
        feature_indices = np.random.choice(num_features, int(np.sqrt(num_features)), replace=False)

        print(feature_indices)
        if isinstance(X, np.ndarray):
            return X[sample_indices][:, feature_indices], Y[sample_indices]
        else:
            return X.iloc[sample_indices, feature_indices], Y.iloc[sample_indices]

    def fit(self, X, Y):
        self.trees = []
        for _ in range(self.n_estimators):
            X_sample, Y_sample = self.bootstrap_sample(X, Y)
            tree = FastDecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )

            tree.fit(X_sample, Y_sample.reshape(-1, 1))
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])])
