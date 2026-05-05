# DBSCAN Methods

## `sklearn.cluster.DBSCAN`

### Purpose
Density-based spatial clustering.

### Syntax
`model = DBSCAN(eps=0.5, min_samples=5)`

### Common Arguments
- `eps` (float): The maximum distance between two samples for them to be considered in the same neighborhood.
- `min_samples` (int): Number of samples required to form a core point.
- `metric` (str): The distance metric to use ('euclidean' is default).

### Common Attributes / Properties
- `labels_`: Cluster labels. Outliers are given the label `-1`.
- `core_sample_indices_`: Indices of core samples.

### Typical Workflow
1. Scale the data (Mandatory).
2. Use NearestNeighbors to plot a K-distance graph to find the optimal `eps`.
3. Fit `DBSCAN`.
4. Separate the `-1` labels to analyze anomalies.
