# Hierarchical Clustering

## Learning Objective
Understand how hierarchical clustering builds a tree of clusters and how to interpret a dendrogram.

## What Is This Topic?
Hierarchical clustering groups data over a variety of scales by creating a cluster tree or dendrogram. It does not require pre-specifying the number of clusters.

## Why This Topic Matters
It provides a visual taxonomy of the data, which is highly interpretable for business stakeholders, especially when identifying sub-groups within groups.

## Core Intuition
**Agglomerative (Bottom-Up):** Start with each point as its own cluster. Find the two closest clusters and merge them. Repeat until everything is one giant cluster.
**Divisive (Top-Down):** Start with all points in one cluster and recursively split them.

## Key Concepts
- Dendrogram
- Linkage Criteria (Ward, Complete, Average, Single)
- Cophenetic Correlation

## Step-by-Step Explanation (Agglomerative)
1. Treat each data point as a single cluster.
2. Compute the distance matrix between all clusters.
3. Merge the two closest clusters.
4. Update the distance matrix.
5. Repeat steps 3 and 4 until only one cluster remains.

## Important Parameters / Options / Settings
- `linkage`: Determines how distance between clusters is defined. 
  - `ward`: Minimizes variance within merged clusters (similar to K-Means).
  - `complete`: Maximum distance between points in clusters.
  - `average`: Average distance between points in clusters.
  - `single`: Minimum distance (prone to chaining).

## Output / Result Interpretation
- Dendrogram: A tree-like diagram. The y-axis represents the distance at which clusters were merged. Cutting the tree at a certain y-value gives you the final clusters.

## Real-World Uses
- Genetic taxonomy (biology).
- Organizing documents into a hierarchy of topics.
- Portfolio diversification in finance.

## Advantages
- No need to guess K in advance.
- Easy to explain via dendrograms.

## Limitations
- Computationally expensive ($O(N^3)$ or $O(N^2 \log N)$); not suitable for huge datasets.
- Cannot undo a merge once it happens.

## Common Mistakes
- Using it on 100,000+ rows (it will crash or take forever).
- Not understanding the difference between linkage methods.

## Related Methods
- K-Means (flat clustering).

## Code References
- `code/example-01-basic-agglomerative.py`
- `code/example-02-dendrogram.py`
- `code/example-03-real-world-taxonomy.py`
