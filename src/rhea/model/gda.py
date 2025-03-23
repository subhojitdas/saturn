import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class GaussianDiscriminantAnalysis:
    def __init__(self):
        self.phi = None  # Class prior P(y = 1)
        self.mu_0 = None  # Mean vector for class 0
        self.mu_1 = None  # Mean vector for class 1
        self.sigma = None  # Shared covariance matrix

    def fit(self, X, y):
        m, n = X.shape

        # Calculate phi (prior probability for class 1)
        self.phi = np.mean(y)

        # Separate class-wise data
        X_0 = X[y == 0]
        X_1 = X[y == 1]

        # Compute the mean vectors for both classes
        self.mu_0 = np.mean(X_0, axis=0)
        self.mu_1 = np.mean(X_1, axis=0)

        # Compute shared covariance matrix
        sigma_0 = np.dot((X_0 - self.mu_0).T, X_0 - self.mu_0)
        sigma_1 = np.dot((X_1 - self.mu_1).T, X_1 - self.mu_1)
        self.sigma = (sigma_0 + sigma_1) / m

    def predict(self, X):
        """Make predictions using GDA."""
        inv_sigma = np.linalg.inv(self.sigma)

        # Compute decision boundary
        p1 = self._gaussian_prob(X, self.mu_1, inv_sigma)
        p0 = self._gaussian_prob(X, self.mu_0, inv_sigma)

        # Apply Bayes Rule
        return (p1 * self.phi) > (p0 * (1 - self.phi))

    def _gaussian_prob(self, X, mu, inv_sigma):
        """Compute Gaussian probability."""
        diff = X - mu
        exponent = -0.5 * np.sum(diff @ inv_sigma * diff, axis=1)
        return np.exp(exponent)


# Generate synthetic data
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=24,
    n_clusters_per_class=1
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train GDA model
gda = GaussianDiscriminantAnalysis()
gda.fit(X_train, y_train)

# Predict on test data
y_pred = gda.predict(X_test)

# Calculate accuracy
print(f"GDA Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")


# Plot decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
    plt.title("Gaussian Discriminant Analysis Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


plot_decision_boundary(gda, X, y)
