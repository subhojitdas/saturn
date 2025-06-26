import numpy as np
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

class RandomForestClassifierCustom:
    def __init__(self, n_estimators=100, max_features="sqrt", max_depth=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.trees = []
        self.feature_indices = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def _select_features(self, X):
        n_features = X.shape[1]
        if self.max_features == "sqrt":
            n_selected = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            n_selected = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            n_selected = self.max_features
        else:
            n_selected = n_features
        selected = np.random.choice(n_features, size=n_selected, replace=False)
        return selected

    def fit(self, X, y):
        self.trees = []
        self.feature_indices = []

        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            selected_features = self._select_features(X_sample)
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_sample[:, selected_features], y_sample)

            self.trees.append(tree)
            self.feature_indices.append(selected_features)

    def predict(self, X):
        all_preds = []
        for tree, features in zip(self.trees, self.feature_indices):
            preds = tree.predict(X[:, features])
            all_preds.append(preds)
        all_preds = np.array(all_preds).T # (n_sample, n_estimator)
        majority_votes = []
        for row in all_preds:
            vote = Counter(row).most_common(1)[0][0]
            majority_votes.append(vote)
        return majority_votes


