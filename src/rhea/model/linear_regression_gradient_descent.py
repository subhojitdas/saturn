import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ssl
from sklearn.datasets import fetch_california_housing

ssl._create_default_https_context = ssl._create_unverified_context
data = fetch_california_housing()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

theta = np.random.randn(X_train.shape[1], 1)
learning_rate = 0.0001
num_iterations = 1000
m = len(X_train)

cost_history = []
for i in range(num_iterations):
    gradients = (2/m) * X_train.T @ (X_train @ theta - y_train.reshape(-1, 1))
    theta -= learning_rate * gradients
    cost = (1/m) * np.sum((X_train @ theta - y_train.reshape(-1, 1)) ** 2)
    cost_history.append(cost)

    if (i + 1) % 50 == 0:
        print(f'Epoch [{i + 1}/{i}], Loss: {cost:.4f}')
        # print(f"Cost at iteration {i}: {cost}, weights: {theta.flatten()}")

# Predictions
y_pred = X_test @ theta

# Print final model weights
print(f"Estimated Weights (theta): {theta.flatten()}")  # [Bias, Coeff1, Coeff2, Coeff3]

# Plot cost function over iterations
plt.plot(range(num_iterations), cost_history, color="blue")
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error (Cost)")
plt.title("Cost Function Convergence")
plt.show()

# Plot actual vs predicted values
plt.scatter(y_test, y_pred, color="blue", alpha=0.6)
plt.xlabel("Actual Target y")
plt.ylabel("Predicted Target y")
plt.title("Linear Regression Predictions (Multiple Features)")
plt.axline([0, 0], [1, 1], color="red", linestyle="--")  # Reference line (y = x)
plt.show()
