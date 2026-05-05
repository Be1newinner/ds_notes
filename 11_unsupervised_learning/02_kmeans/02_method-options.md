# K-Means Methods and Properties

## `sklearn.cluster.KMeans`

### Purpose
Partitions data into K clusters.

### Syntax
`model = KMeans(n_clusters=3, random_state=42)`

### Common Arguments
- `n_clusters` (int): Number of clusters to form.
- `init` (str): 'k-means++' (default) or 'random'.
- `n_init` (int): Number of times the algorithm will be run with different centroid seeds.
- `random_state` (int): Random number generation for centroid initialization.

### Common Attributes / Properties
- `cluster_centers_`: Coordinates of cluster centers.
- `labels_`: Labels of each point.
- `inertia_`: Sum of squared distances of samples to their closest cluster center.

### Output / Return Type
The fitted model object containing the properties above.

### Typical Workflow
1. Preprocess and scale data (`StandardScaler`).
2. Run a loop to test different values of K.
3. Plot Inertia (Elbow Method) and Silhouette Scores to choose K.
4. Fit final `KMeans` with chosen K.
5. Analyze `cluster_centers_`.

## `sklearn.metrics.silhouette_score`
### Purpose
Evaluates the quality of clusters (higher is better, range -1 to 1).
### Syntax
`score = silhouette_score(X, labels)`
