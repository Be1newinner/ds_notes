# Clustering Fundamentals

## Learning Objective
Understand what unsupervised learning is, the goals of clustering, and the fundamental concept of distance metrics.

## What Is This Topic?
Clustering is the task of dividing the population or data points into a number of groups such that data points in the same groups are more similar to other data points in the same group than those in other groups.

## Why This Topic Matters
Before learning algorithms like K-Means, we must define what "similarity" means and understand that there is no "ground truth" to check our answers against.

## Core Intuition
Tell me who your friends are, and I will tell you who you are. We measure "closeness" between data points using math (distance) and group the closest ones together.

## Key Concepts
- Unsupervised Learning vs Supervised Learning
- Hard Clustering vs Soft Clustering
- Distance Metrics (Euclidean, Manhattan, Cosine)

## Step-by-Step Explanation
1. We have a dataset with features but no labels.
2. We define a way to measure the distance between any two data points.
3. We use an algorithm to group points such that intra-cluster distance is minimized and inter-cluster distance is maximized.
4. We assign meaning (labels) to the clusters based on their characteristics.

## Important Parameters / Options / Settings
- Distance Metric: Euclidean (straight line), Manhattan (city block), Cosine (angle).

## Output / Result Interpretation
The output is simply a cluster ID for each row in the dataset.

## Real-World Uses
- Market Segmentation
- Social Network Analysis

## Advantages
- Does not require labeled data (which is expensive to get).
- Helps discover hidden patterns.

## Limitations
- Highly subjective; evaluation is difficult.
- Sensitive to scaling of features.

## Common Mistakes
- Not scaling data before calculating distances.
- Trying to force clustering when data has no inherent groups.

## Related Methods
- Dimensionality Reduction (PCA)

## Code References
- `code/example-01-distances.py`
