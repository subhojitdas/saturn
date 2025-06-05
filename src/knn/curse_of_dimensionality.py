import numpy as np
import matplotlib.pyplot as plt


def simulate_knn_distances(num_points=1000, max_dim=100):
    dims = np.arange(1, max_dim + 1)
    nearest_dists = []
    farthest_dists = []
    ratios = []

    for d in dims:
        # points are chosen uniformly
        points = np.random.rand(num_points, d)

        # p = 2 in (Sum(| x1 - x2 |)**p)**(1/p)
        distances = np.linalg.norm(points, axis=1)

        nearest = np.min(distances)
        farthest = np.max(distances)
        ratio = nearest / farthest

        nearest_dists.append(nearest)
        farthest_dists.append(farthest)
        ratios.append(ratio)

    return dims, nearest_dists, farthest_dists, ratios


dims, nearest_dists, farthest_dists, ratios = simulate_knn_distances()

plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
plt.plot(dims, nearest_dists, label='Nearest Distance')
plt.xlabel("Dimensions")
plt.ylabel("Distance")
plt.title("Nearest Neighbor Distance")
plt.grid()

plt.subplot(1, 3, 2)
plt.plot(dims, farthest_dists, label='Farthest Distance', color='orange')
plt.xlabel("Dimensions")
plt.ylabel("Distance")
plt.title("Farthest Neighbor Distance")
plt.grid()

plt.subplot(1, 3, 3)
plt.plot(dims, ratios, label='Nearest / Farthest', color='green')
plt.xlabel("Dimensions")
plt.ylabel("Ratio")
plt.title("Distance Ratio: Nearest / Farthest")
plt.grid()

plt.tight_layout()
plt.show()
