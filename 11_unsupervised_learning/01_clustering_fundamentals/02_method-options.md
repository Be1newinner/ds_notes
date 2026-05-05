# Clustering Distance Metrics

## `sklearn.metrics.pairwise`

### Purpose
Calculates distances between data points.

### Syntax
`euclidean_distances(X, Y)`
`cosine_distances(X, Y)`
`manhattan_distances(X, Y)`

### Common Arguments
- `X`: Array of points.
- `Y`: Optional array of points. If None, computes distance between all pairs in X.

### Return Type
A distance matrix (NumPy array).

### Typical Workflow
1. Scale data.
2. Compute distances to understand similarities before clustering.
