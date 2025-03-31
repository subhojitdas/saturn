import numpy as np

class CustomSVM:
    def __init__(self, C=1.0, kernel='linear', degree=3, gamma=0.5, max_iter=1000, learning_rate=0.001):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.w = None
        self.b = 0

    def _kernel_function(self, x1, x2):
        """Compute kernel based on user selection"""
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'poly':
            return (np.dot(x1, x2) + 1) ** self.degree
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        else:
            raise ValueError("Unknown kernel. Choose from ['linear', 'poly', 'rbf']")

    def fit(self, X, y):
        """Train SVM using gradient descent"""
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.max_iter):
            for i in range(n_samples):
                if y[i] * (np.dot(X[i], self.w) + self.b) < 1:
                    # Misclassified - apply hinge loss gradient
                    self.w -= self.learning_rate * (self.w - self.C * y[i] * X[i])
                    self.b -= self.learning_rate * (-self.C * y[i])
                else:
                    # Correctly classified - only apply regularization
                    self.w -= self.learning_rate * self.w

    def predict(self, X):
        """Predict labels using trained SVM"""
        predictions = np.dot(X, self.w) + self.b
        return np.sign(predictions)

    def score(self, X, y):
        """Evaluate accuracy"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
