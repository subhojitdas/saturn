import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape

        self.theta = np.zeros(n + 1)
        X_bias = np.c_[np.ones((m, 1)), X]

        for i in range(self.n_iterations):
            z = np.dot(X_bias, self.theta)
            h = self.sigmoid(z)

            gradient = (1 / m) * np.dot(X_bias.T, (h - y))

            self.theta -= self.learning_rate * gradient

    def predict_probability(self, X):
        m, n = X.shape
        X_bias = np.c_[np.ones((m, 1)), X]
        z = np.dot(X_bias, self.theta)
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_probability(X)
        return (probabilities >= threshold).astype(int)

    def accuracy(self, y_true, y_pred):
        """Calculate accuracy."""
        return np.mean(y_true == y_pred)

    def plot_data(self, X, y):
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Data Points Visualization")
        plt.legend()
        plt.show()

    def plot_decision_boundary(self, X, y):
        """Plot the decision boundary for 2D data."""
        # Plot data points
        self.plot_data(X, y)

        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))

        # Flatten and predict on the grid
        grid = np.c_[np.ones((xx.ravel().shape[0], 1)), xx.ravel(), yy.ravel()]  # Add bias term
        Z = self.sigmoid(np.dot(grid, self.theta)).reshape(xx.shape)

        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Decision Boundary")
        plt.legend()
        plt.show()


X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=24
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(n_iterations=1000, learning_rate=0.01)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = model.accuracy(y_test, y_pred)
print(accuracy)

model.plot_data(X_train, y_train)
model.plot_decision_boundary(X_train, y_train)
