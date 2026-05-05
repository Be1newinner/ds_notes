import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Generate stretched data
np.random.seed(42)
X, y_true = make_blobs(n_samples=400, centers=3, cluster_std=0.5, random_state=42)
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_stretched = np.dot(X, transformation)

# Fit GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
labels = gmm.fit_predict(X_stretched)

plt.scatter(X_stretched[:, 0], X_stretched[:, 1], c=labels, cmap='viridis')
plt.title("GMM Clustering (Handles Elliptical Shapes)")
plt.show()
