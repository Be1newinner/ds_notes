# PCA Method and Options

## Scikit-Learn: `sklearn.decomposition.PCA`

### Purpose
To perform linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.

### Syntax
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
```

### Common Arguments
- `n_components` (int, float, or None): Number of components to keep.
  - If `int` (e.g., `2`): Keeps exactly 2 components.
  - If `float` (e.g., `0.95`): Keeps enough components to explain 95% of the variance.
  - If `None`: Keeps all components (useful for looking at the cumulative variance plot).
- `whiten` (bool): When True, it scales the components to have unit variance. Can sometimes improve predictive accuracy of downstream models.
- `random_state` (int): Seed for reproducibility when using randomized solvers.

### Common Methods
- `fit(X)`: Learns the principal components from the training data X.
- `transform(X)`: Applies the dimensionality reduction to X, returning the new, lower-dimensional data.
- `fit_transform(X)`: Fits the model and transforms the data in one step (more efficient for training data).
- `inverse_transform(X_transformed)`: Transforms data back to its original space (useful for checking compression loss).

### Common Attributes
- `components_`: The principal axes in feature space, representing the directions of maximum variance (the eigenvectors). Shape is `(n_components, n_features)`.
- `explained_variance_`: The amount of variance explained by each of the selected components (the eigenvalues).
- `explained_variance_ratio_`: Percentage of variance explained by each component. Very useful for plotting the "Scree Plot".

### Typical Workflow
1. Import `StandardScaler` and `PCA`.
2. Fit and transform the data with `StandardScaler`.
3. Initialize `PCA` with desired components.
4. Call `fit_transform()` on the scaled data.
5. Check `explained_variance_ratio_.sum()` to see total information retained.

### Common Mistakes
- **Forgetting `StandardScaler`**: This is the #1 mistake. PCA maximizes variance, so a feature measured in millions will dominate a feature measured in decimals if not scaled.
- **Fitting PCA on Test Data**: Always `fit` PCA on the training set, and only `transform` the test set to avoid data leakage.
