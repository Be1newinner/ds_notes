import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)

# To find eps, we look at the distance to the Kth nearest neighbor
# If min_samples = 5, we look at the 5th nearest neighbor
k = 5
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(X)
distances, indices = neighbors_fit.kneighbors(X)

# Sort distances
distances = np.sort(distances[:, k-1], axis=0)

plt.plot(distances)
plt.title("K-Distance Graph to find optimal eps")
plt.xlabel("Points sorted by distance")
plt.ylabel(f"{k}th Nearest Neighbor Distance")
plt.grid(True)
plt.show()
# Look for the "elbow" or sharp curve upward to pick eps
