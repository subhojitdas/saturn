import numpy as np
from sklearn.datasets import make_moons
from sklearn.svm import SVC
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

model = SVC(kernel='rbf', C=1, gamma='scale')
model.fit(X, y)

# 🎯 Plot decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.coolwarm)
    plt.title("SVM with RBF Kernel")
    plt.show()

# 📊 Visualize
plot_decision_boundary(model, X, y)
