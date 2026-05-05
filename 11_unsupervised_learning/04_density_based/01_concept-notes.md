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
