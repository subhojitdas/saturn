from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_blobs
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

# dbscan = DBSCAN(eps=0.2, min_samples=5)
# labels = dbscan.fit_predict(X)
#
# plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='Spectral', s=50)
# plt.show()


def euclidian_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def region_query(X, point_idx, epsilon):
    neighbors = []
    for idx, point in enumerate(X):
        if euclidian_distance(point, X[point_idx]) < epsilon:
            neighbors.append(idx)
    return neighbors

def expand_cluster(X, labels, point_idx, cluster_id, epsilon, min_samples):
    seeds = region_query(X, point_idx, epsilon)
    if len(seeds) < min_samples:
        labels[point_idx] = -1
        return False
    else:
        labels[point_idx] = cluster_id
        queue = deque(seeds)
        while queue:
            current_point = queue.popleft()
            if labels[current_point] == -1:
                labels[current_point] = cluster_id
            if labels[current_point] != 0:
                continue
            labels[current_point] = cluster_id
            nei = region_query(X, current_point, epsilon)
            if len(nei) >= min_samples:
                queue.extend(nei)
    return True

def dbscan(X, epsilon, min_samples):
    labels = np.zeros(X.shape[0], dtype=int)
    cluster_id = 0
    for point_idx in range(X.shape[0]):
        if labels[point_idx] != 0:
            continue
        expanded = expand_cluster(X, labels, point_idx, cluster_id + 1, epsilon, min_samples)
        if expanded:
            print(f"cluster {cluster_id + 1} done")
            cluster_id += 1

    return labels

epsilon = 1.2
min_samples = 5

# X, _ = make_moons(n_samples=100, noise=0.05, random_state=18)
X, _ = make_blobs(n_samples=500, centers=7, cluster_std=1.0, random_state=18)
labels = dbscan(X, epsilon, min_samples)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='Spectral', s=50)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.show()




# print(euclidian_distance(np.array([0, 1]), np.array([1, 0])))