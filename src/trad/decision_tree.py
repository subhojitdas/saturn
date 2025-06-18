import numpy as np

def gini_impurity(y):
    """
    Gini Impurity => If we randomly pick two items with replacement from a node , what is the probability they belong to diff classes
    probability of choosing two items same class (pi) with same label sum_over_i(pi*pi)
    probability of choosing two items diff class = 1 - sum_over_i(pi*pi)
    """
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)


def dataset_split(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]


def best_split(X, y):
    best_gini = float("inf")
    best_index, best_threshold = None, None

    n_samples, n_features = X.shape
    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            X_left, y_left, X_right, y_right = dataset_split(X, y, feature_index, threshold)

            if len(y_right) == 0 or len(y_left) == 0:
                continue
            gini_right = gini_impurity(y_right)
            gini_left = gini_impurity(y_left)
            weighted_gini = (gini_left * len(y_left) + gini_right * len(y_right)) / len(y)
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_index = feature_index
                best_threshold = threshold
    return best_index, best_threshold






X = np.array([
    [2.7, 2.5],
    [1.3, 1.5],
    [3.0, 3.5],
    [0.5, 0.7],
    [3.2, 3.0],
])

y = np.array([1, 0, 1, 0, 1])

gini = gini_impurity(y)
print(gini)