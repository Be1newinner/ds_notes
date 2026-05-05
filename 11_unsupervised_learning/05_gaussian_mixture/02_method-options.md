# Gaussian Mixture Methods

## `sklearn.mixture.GaussianMixture`

### Purpose
Fits a Gaussian mixture model.

### Syntax
`model = GaussianMixture(n_components=3, covariance_type='full')`

### Common Arguments
- `n_components` (int): Number of mixture components.
- `covariance_type` (str): 'full', 'tied', 'diag', 'spherical'.
- `init_params` (str): Method used to initialize weights ('kmeans' is default).

### Common Attributes / Properties
- `means_`: Means of each mixture component.
- `covariances_`: Covariances of each mixture component.
- `weights_`: The mixing weights.

### Output / Return Type
Fitted model object.

### Typical Workflow
1. Scale data.
2. Fit GMMs with varying `n_components`.
3. Extract `model.bic(X)` to find the optimal number of components.
4. Fit final model.
5. Use `model.predict_proba(X)` to get soft assignments.
