# K-Means and Variants

## Learning Objective
Understand the K-Means algorithm, how to find the optimal number of clusters, and how to evaluate clustering performance.

## What Is This Topic?
K-Means is a partitional clustering algorithm that divides data into K distinct, non-overlapping clusters by minimizing the variance within each cluster.

## Why This Topic Matters
It is the most widely used clustering algorithm in the industry due to its simplicity and speed.

## Core Intuition
1. Guess the center of the clusters.
2. Assign points to the nearest center.
3. Move the center to the average of the points assigned to it.
4. Repeat until the centers stop moving.

## Key Concepts
- Centroid
- Inertia (Within-Cluster Sum of Squares)
- The Elbow Method
- Silhouette Score
- K-Means++ Initialization

## Step-by-Step Explanation
1. Specify the desired number of clusters K.
2. Randomly select K data points to be the initial centroids.
3. Assign each data point to the closest centroid.
4. Recompute the centroids by taking the mean of all data points in the cluster.
5. Repeat steps 3 and 4 until convergence.

## Important Parameters / Options / Settings
- `n_clusters`: The number of clusters (K).
- `init`: Usually 'k-means++' to speed up convergence and avoid bad local minima.
- `max_iter`: Maximum number of iterations.
- `random_state`: For reproducibility.

## Output / Result Interpretation
- Cluster labels: A number from 0 to K-1 for each point.
- Cluster centers: The coordinates representing the "average" profile of the cluster.

## Real-World Uses
- Customer Segmentation
- Document Classification
- Image Color Quantization

## Advantages
- Fast and scalable to large datasets.
- Easy to understand and interpret.

## Limitations
- Must choose K manually.
- Assumes clusters are spherical and of similar size.
- Sensitive to outliers.

## Common Mistakes
- Forgetting to scale the data.
- Not trying different values of K.

## Related Methods
- K-Medoids (robust to outliers).
- Mini-Batch K-Means (for massive datasets).

## Code References
- `code/example-01-basic-kmeans.py`
- `code/example-02-elbow-silhouette.py`
- `code/example-03-customer-segmentation.py`
