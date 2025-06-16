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