# Density-Based Clustering (DBSCAN)

## Learning Objective
Understand how density-based algorithms work, why they are powerful for irregularly shaped clusters, and how they identify anomalies.

## What Is This Topic?
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups together points that are closely packed together, while marking points in low-density regions as outliers (noise).

## Why This Topic Matters
Real-world data is rarely grouped in perfect spheres (like K-Means assumes). DBSCAN adapts to the shape of the data and automatically finds anomalies.

## Core Intuition
If a data point has many close neighbors, it is part of a cluster. If a point is alone in the middle of nowhere, it is an outlier.

## Key Concepts
- **Core Point:** A point that has at least `min_samples` points within `eps` distance.
- **Border Point:** A point within `eps` of a core point, but has fewer than `min_samples` neighbors.
- **Noise (Outlier):** A point that is neither a core nor a border point.

## Step-by-Step Explanation
1. Pick a random unvisited point.
2. Count how many points are within a radius of `eps`.
3. If the count $\ge$ `min_samples`, start a new cluster and add all these points to a queue.
4. Expand the cluster by checking the neighbors of the points in the queue.
5. If the count < `min_samples`, label the point as Noise.
6. Repeat until all points are visited.

## Important Parameters / Options / Settings
- `eps` ($\epsilon$): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
- `min_samples`: The number of samples in a neighborhood for a point to be considered as a core point.

## Output / Result Interpretation
- Points assigned a cluster label $0, 1, 2, ...$
- Points assigned a label of `-1` are outliers/noise.

## Real-World Uses
- Fraud detection (anomalies).
- Geographic clustering (e.g., grouping earthquake locations).

## Advantages
- Does not require specifying the number of clusters.
- Can find arbitrarily shaped clusters.
- Robust to outliers (and identifies them).

## Limitations
- Struggles with clusters of varying densities.
- Highly sensitive to the `eps` and `min_samples` parameters.

## Common Mistakes
- Not scaling data. `eps` relies completely on distance. If unscaled, `eps` is meaningless.
- Guessing `eps` instead of using a K-distance graph to estimate it.

## Related Methods
- OPTICS (handles varying densities better than DBSCAN).
- Isolation Forest (specifically for anomaly detection).

## Code References
- `code/example-01-basic-dbscan.py`
- `code/example-02-hyperparameter-tuning.py`
- `code/example-03-real-world-outliers.py`


---

## DBSCAN Methods

### `sklearn.cluster.DBSCAN`

#### Purpose
Density-based spatial clustering.

#### Syntax
`model = DBSCAN(eps=0.5, min_samples=5)`

#### Common Arguments
- `eps` (float): The maximum distance between two samples for them to be considered in the same neighborhood.
- `min_samples` (int): Number of samples required to form a core point.
- `metric` (str): The distance metric to use ('euclidean' is default).

#### Common Attributes / Properties
- `labels_`: Cluster labels. Outliers are given the label `-1`.
- `core_sample_indices_`: Indices of core samples.

#### Typical Workflow
1. Scale the data (Mandatory).
2. Use NearestNeighbors to plot a K-distance graph to find the optimal `eps`.
3. Fit `DBSCAN`.
4. Separate the `-1` labels to analyze anomalies.

---

## Examples: DBSCAN

### Code References
- `code/example-01-basic-dbscan.py` — how DBSCAN handles non-linear datasets like make_moons.
- `code/example-02-hyperparameter-tuning.py` — using NearestNeighbors to find the optimal `eps`.
- `code/example-03-real-world-outliers.py` — using DBSCAN specifically for anomaly detection.

---

## Practice: DBSCAN
1. In Example 3, change `min_samples` to 5. What happens to the clustering? Why?
2. Generate a dataset with two circles (using `make_circles` from sklearn). Apply KMeans and DBSCAN. Compare the plots.

---

## Interview Questions: DBSCAN
1. How does DBSCAN differentiate between a core point, border point, and noise?
2. Why is DBSCAN considered better than K-Means for anomaly detection?
3. What happens to DBSCAN results if the scale of your features is drastically different (e.g., age in years vs income in millions)?
4. What is a weakness of DBSCAN compared to K-Means?

---

## Python Code Examples

### `example-01-basic-dbscan.py`

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# K-Means fails on this dataset, DBSCAN succeeds
X, y = make_moons(n_samples=300, noise=0.08, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.title("DBSCAN Clustering (handles non-linear shapes)")
plt.show()
```

### `example-02-hyperparameter-tuning.py`

```python
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
```

### `example-03-real-world-outliers.py`

```python
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Network traffic data
data = {
    'Bytes_Sent': [500, 600, 550, 480, 520, 10000, 490, 510, 530, 2],
    'Packets_Sent': [10, 12, 11, 9, 10, 200, 9, 10, 11, 1]
}
df = pd.DataFrame(data)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# We want to identify the massive spike (10000) and tiny drop (2) as anomalies
dbscan = DBSCAN(eps=0.5, min_samples=3)
df['Cluster'] = dbscan.fit_predict(X_scaled)

print("Network Traffic Analysis:")
print(df)
print("\nOutliers detected (Cluster == -1):")
print(df[df['Cluster'] == -1])
```
