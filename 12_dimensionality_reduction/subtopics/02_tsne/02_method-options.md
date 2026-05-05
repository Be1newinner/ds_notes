# t-SNE Method and Options

## Scikit-Learn: `sklearn.manifold.TSNE`

### Purpose
To reduce dimensionality for the sake of visualizing high-dimensional data in 2D or 3D.

### Syntax
```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30.0, random_state=42)
```

### Common Arguments
- `n_components` (int): Dimension of the embedded space (usually 2 or 3).
- `perplexity` (float): Relates to the number of nearest neighbors that is used in other manifold learning algorithms. Consider selecting a value between 5 and 50. Different values can result in significantly different plots.
- `n_iter` (int): Maximum number of iterations for the optimization. Should be at least 250.
- `random_state` (int): Seed for reproducibility. Highly recommended because t-SNE is stochastic.

### Common Methods
- `fit_transform(X)`: Fits X into an embedded space and returns that transformed output.
- **Note**: Unlike PCA, there is NO `transform(X)` method. You cannot fit t-SNE on training data and then transform test data.

### Typical Workflow
1. Standardize the data using `StandardScaler`.
2. Initialize `TSNE` with `n_components=2` and a specific `perplexity`.
3. Use `fit_transform()` on the data.
4. Plot the resulting 2D array using matplotlib/seaborn, coloring points by their class label.

### Common Mistakes
- **Assuming t-SNE is a preprocessing step for ML models**: Because there is no `transform` method, you cannot use it in a standard train/test pipeline. It is an Exploratory Data Analysis (EDA) tool.
- **Ignoring Perplexity**: The default perplexity might not suit your data size. You should try multiple values (e.g., 5, 30, 50).
