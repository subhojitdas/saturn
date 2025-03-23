import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define mean and covariance matrix for 3D multivariate normal
mean = [0, 0, 2]  # Centered at origin
cov = [[1, 0.05, 0.3],
       [0.05, 1, 0.2],
       [0.3, 0.2, 1]]  # Covariance matrix

# Generate grid for 3D plot
x, y = np.mgrid[-3:3:.1, -3:3:.1]
pos = np.empty(x.shape + (3,))
pos[:, :, 0] = x
pos[:, :, 1] = y
pos[:, :, 2] = 0  # Slice along Z-axis

# Create multivariate normal distribution
rv = multivariate_normal(mean, cov)

# Evaluate the PDF on the grid
pdf_values = rv.pdf(pos)

# Scale the PDF values to enhance visibility
scaling_factor = 50  # Increase this to exaggerate the bump
pdf_values_scaled = pdf_values * scaling_factor

# Plotting the PDF as a 3D surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot PDF as a 3D surface
surf = ax.plot_surface(x, y, pdf_values_scaled, cmap='viridis', alpha=0.8)
fig.colorbar(surf, shrink=0.5, aspect=5, label='Scaled PDF Density')

# Labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Scaled PDF')
ax.set_title('3D Visualization of Scaled Multivariate Normal Distribution')

plt.show()
