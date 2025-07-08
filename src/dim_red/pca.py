import numpy as np
import matplotlib.pyplot as plt

np.random.seed(18)

mean = [0, 0]
cov_mat = [[3, 2], [2, 1]]

X = np.random.multivariate_normal(mean=mean, cov=cov_mat, size=300)

X_meaned = X - np.mean(X, axis=0)
cov_mat = np.cov(X_meaned.T)
eigen_values, eigenvectors = np.linalg.eigh(cov_mat)

sorted_indices = np.argsort(eigen_values)[::-1]
eigen_values = eigen_values[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

k = 1
topk_eigenvectors = eigenvectors[:, :k]
X_pca = X_meaned @ topk_eigenvectors
X_reconstructed = X_pca @ topk_eigenvectors.T + np.mean(X, axis=0)



# ----- plot ------
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], label='PCA Projection (1D)', alpha=0.8, marker='x')
for orig, recon in zip(X, X_reconstructed):
    plt.plot([orig[0], recon[0]], [orig[1], recon[1]], 'k--', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()