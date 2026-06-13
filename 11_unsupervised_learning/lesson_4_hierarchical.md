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


---

## Hierarchical Clustering Methods

### `sklearn.cluster.AgglomerativeClustering`

#### Purpose
Performs hierarchical clustering using a bottom-up approach.

#### Syntax
`model = AgglomerativeClustering(n_clusters=3, linkage='ward')`

#### Common Arguments
- `n_clusters` (int): Number of clusters to find.
- `distance_threshold` (float): The linkage distance threshold above which clusters will not be merged (use instead of `n_clusters`).
- `linkage` (str): 'ward', 'complete', 'average', 'single'.

#### Common Attributes / Properties
- `labels_`: Cluster labels.

#### Output / Return Type
Fitted model.

### `scipy.cluster.hierarchy`

#### Purpose
Used to create and plot dendrograms. (Scikit-learn does not have a built-in dendrogram plotting function).

#### Syntax
```python
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(X, 'ward')
dendrogram(Z)
```

---

## Examples: Hierarchical Clustering

### Code References
- `code/example-01-basic-agglomerative.py` — simple clustering on moon-shaped data.
- `code/example-02-dendrogram.py` — how to generate and read a dendrogram using SciPy.
- `code/example-03-real-world-taxonomy.py` — practical example of grouping products.

---

## Practice: Hierarchical Clustering
1. Generate a dataset using `make_blobs`. Plot the dendrogram using 'single', 'average', and 'ward' linkage. Observe the differences.
2. In example 3, change `n_clusters` to `None` and set `distance_threshold=3.0`. How many clusters are formed?

---

## Interview Questions: Hierarchical Clustering
1. What is the difference between K-Means and Hierarchical Clustering?
2. What is a dendrogram and how do you read it?
3. Explain the difference between single, complete, and ward linkage.
4. Why might you avoid Hierarchical clustering on a dataset with 1 million rows?

---

## Python Code Examples

### `example-01-basic-agglomerative.py`

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Generate data
X, _ = make_moons(n_samples=200, noise=0.05, random_state=0)

# Fit model
agg_cluster = AgglomerativeClustering(n_clusters=2, linkage='single')
labels = agg_cluster.fit_predict(X)

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title("Agglomerative Clustering (Single Linkage)")
plt.show()
```

### `example-02-dendrogram.py`

```python
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
```

### `example-03-real-world-taxonomy.py`

```python
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Product categories dataset
data = {
    'Product': ['Laptop', 'Desktop', 'Tablet', 'Smartphone', 'Smartwatch', 'Headphones', 'Speaker'],
    'Price': [1200, 1500, 400, 800, 250, 150, 100],
    'Portability': [7, 1, 9, 10, 10, 8, 5],
    'Processing_Power': [8, 10, 5, 7, 3, 1, 1]
}
df = pd.DataFrame(data)

# Scale features
X = df[['Price', 'Portability', 'Processing_Power']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit clustering to find 3 groups
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
df['Cluster'] = model.fit_predict(X_scaled)

print(df.sort_values('Cluster'))
```
