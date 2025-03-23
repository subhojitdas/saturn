import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class MultiClassGDA:
    def __init__(self):
        self.classes = None
        self.phi = {}  # Class priors
        self.mu = {}  # Mean vectors for each class
        self.sigma = None  # Shared covariance matrix

    def fit(self, X, y):
        """Fit GDA model to training data."""
        m, n = X.shape
        self.classes = np.unique(y)
        num_classes = len(self.classes)

        # Initialize covariance matrix
        sigma_sum = np.zeros((n, n))

        for c in self.classes:
            X_c = X[y == c]
            self.phi[c] = len(X_c) / m  # Prior for class c
            self.mu[c] = np.mean(X_c, axis=0)  # Mean vector for class c
            # Accumulate covariance
            sigma_sum += np.dot((X_c - self.mu[c]).T, X_c - self.mu[c])

        # Compute shared covariance matrix
        self.sigma = sigma_sum / m

    def predict(self, X):
        """Make predictions using multi-class GDA."""
        inv_sigma = np.linalg.inv(self.sigma)
        scores = np.zeros((X.shape[0], len(self.classes)))

        # Calculate Gaussian likelihood for each class
        for i, c in enumerate(self.classes):
            scores[:, i] = self._gaussian_prob(X, self.mu[c], inv_sigma) * self.phi[c]

        # Assign class with maximum posterior probability
        return self.classes[np.argmax(scores, axis=1)]

    def _gaussian_prob(self, X, mu, inv_sigma):
        """Compute Gaussian probability."""
        diff = X - mu
        exponent = -0.5 * np.sum(diff @ inv_sigma * diff, axis=1)
        return np.exp(exponent)

X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=3,
    random_state=42,
    n_clusters_per_class=1
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Multi-Class GDA model
gda = MultiClassGDA()
gda.fit(X_train, y_train)

# Predict on test data
y_pred = gda.predict(X_test)

# Calculate accuracy
print(f"Multi-Class GDA Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")


# Plot decision boundary for multi-class
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
    legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.gca().add_artist(legend1)
    plt.title("Multi-Class Gaussian Discriminant Analysis Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


plot_decision_boundary(gda, X, y)
