
import ssl
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context
data = fetch_california_housing()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def with_normal_equation():
    theta_best = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    y_pred = X_test @ theta_best

    print(f"Estimated Weights (theta): {theta_best.flatten()}")

    plt.scatter(y_test, y_pred, color="blue", alpha=0.6)
    plt.xlabel("Actual Target y")
    plt.ylabel("Predicted Target y")
    plt.title("Linear Regression Predictions (Multiple Features)")
    plt.axline([0, 0], [1, 1], color="red", linestyle="--")  # Reference line (y = x)
    plt.show()


with_normal_equation()
