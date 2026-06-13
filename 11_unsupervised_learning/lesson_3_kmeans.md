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


---

## K-Means Methods and Properties

### `sklearn.cluster.KMeans`

#### Purpose
Partitions data into K clusters.

#### Syntax
`model = KMeans(n_clusters=3, random_state=42)`

#### Common Arguments
- `n_clusters` (int): Number of clusters to form.
- `init` (str): 'k-means++' (default) or 'random'.
- `n_init` (int): Number of times the algorithm will be run with different centroid seeds.
- `random_state` (int): Random number generation for centroid initialization.

#### Common Attributes / Properties
- `cluster_centers_`: Coordinates of cluster centers.
- `labels_`: Labels of each point.
- `inertia_`: Sum of squared distances of samples to their closest cluster center.

#### Output / Return Type
The fitted model object containing the properties above.

#### Typical Workflow
1. Preprocess and scale data (`StandardScaler`).
2. Run a loop to test different values of K.
3. Plot Inertia (Elbow Method) and Silhouette Scores to choose K.
4. Fit final `KMeans` with chosen K.
5. Analyze `cluster_centers_`.

### `sklearn.metrics.silhouette_score`
#### Purpose
Evaluates the quality of clusters (higher is better, range -1 to 1).
#### Syntax
`score = silhouette_score(X, labels)`

---

## Examples: K-Means

### Code References
- `code/example-01-basic-kmeans.py` — simple introduction example.
- `code/example-02-elbow-silhouette.py` — how to find the optimal K using Inertia and Silhouette score.
- `code/example-03-customer-segmentation.py` — practical dataset example with scaling and interpreting centers.

---

## Practice: K-Means
1. Load the Iris dataset, drop the labels, and run K-Means with K=3. Compare the resulting clusters to the actual species.
2. Build a loop to find the best K for the Iris dataset using the Elbow method. Does it match K=3?
3. What happens if you don't scale the Annual Spend vs Purchase Frequency in Example 3? Try it.

---

## Interview Questions: K-Means
1. How does the K-Means algorithm work step-by-step?
2. What is K-Means++ and why is it better than random initialization?
3. How do you decide the optimal number of clusters?
4. What is Inertia? What is Silhouette Score?
5. Can K-Means handle outliers well? Why or why not?

---

## Python Code Examples

### `example-01-basic-kmeans.py`

```python
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
```

### `example-02-elbow-silhouette.py`

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=500, centers=5, cluster_std=0.8, random_state=42)

inertia = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

fig, ax1 = plt.subplots()

ax1.plot(K_range, inertia, 'bo-')
ax1.set_xlabel('Number of clusters (K)')
ax1.set_ylabel('Inertia', color='b')

ax2 = ax1.twinx()
ax2.plot(K_range, silhouette_scores, 'rs-')
ax2.set_ylabel('Silhouette Score', color='r')

plt.title("Elbow Method & Silhouette Score")
plt.show()
```

### `example-03-customer-segmentation.py`

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Simulated e-commerce customer data
data = {
    'Annual_Spend': [1500, 200, 3000, 250, 1800, 4000, 150, 2800],
    'Purchase_Frequency': [12, 2, 24, 3, 15, 30, 1, 20]
}
df = pd.DataFrame(data)

# Scaling is critical!
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Fit KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
df['Cluster'] = kmeans.fit_predict(df_scaled)

print("Segmented Customers:")
print(df)

# Analyze the centers (in original scale)
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(centers_original, columns=['Annual_Spend', 'Purchase_Frequency'])
centers_df.index.name = 'Cluster'
print("\nCluster Profiles (Averages):")
print(centers_df)
```
