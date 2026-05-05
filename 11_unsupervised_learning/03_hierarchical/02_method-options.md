# Hierarchical Clustering Methods

## `sklearn.cluster.AgglomerativeClustering`

### Purpose
Performs hierarchical clustering using a bottom-up approach.

### Syntax
`model = AgglomerativeClustering(n_clusters=3, linkage='ward')`

### Common Arguments
- `n_clusters` (int): Number of clusters to find.
- `distance_threshold` (float): The linkage distance threshold above which clusters will not be merged (use instead of `n_clusters`).
- `linkage` (str): 'ward', 'complete', 'average', 'single'.

### Common Attributes / Properties
- `labels_`: Cluster labels.

### Output / Return Type
Fitted model.

## `scipy.cluster.hierarchy`

### Purpose
Used to create and plot dendrograms. (Scikit-learn does not have a built-in dendrogram plotting function).

### Syntax
```python
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(X, 'ward')
dendrogram(Z)
```
