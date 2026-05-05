import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Create and fit the model
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
kmeans.fit(X)

# Get predictions and centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.7, marker='X')
plt.title("Basic K-Means Clustering")
plt.show()
