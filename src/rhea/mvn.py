import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define mean and covariance matrix
mean = [0, 0, 0]  # Mean vector
cov = [[1, 0.5, 0.3],
       [0.5, 1, 0.2],
       [0.3, 0.2, 1]]  # Covariance matrix

# Generate random samples from the multivariate normal distribution
num_samples = 1000
samples = np.random.multivariate_normal(mean, cov, size=num_samples)

# Plot the 3D scatter plot of the samples
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Extract X, Y, Z coordinates
x, y, z = samples[:, 0], samples[:, 1], samples[:, 2]

# Plot the samples
scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=5, alpha=0.5)

# Add color bar for density
cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Density')

# Set axis labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Visualization of Multivariate Normal Distribution')

plt.show()