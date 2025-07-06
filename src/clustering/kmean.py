from sklearn.cluster import KMeans
import numpy as np


X = np.array([
    [1, 2], [1, 4], [1, 0],
    [10, 2], [10, 4], [10, 0]
])

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

print("Cluster centers:\n", kmeans.cluster_centers_)
print("Labels:\n", kmeans.labels_)