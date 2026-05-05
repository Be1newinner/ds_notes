import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Small dataset for clear dendrogram
np.random.seed(42)
X = np.random.rand(15, 2)

# Generate the linkage matrix
Z = linkage(X, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
dendrogram(Z)
plt.axhline(y=0.5, color='r', linestyle='--') # Example of "cutting" the tree
plt.show()
