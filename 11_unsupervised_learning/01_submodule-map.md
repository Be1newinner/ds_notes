# Submodule Map: Unsupervised Learning

## 1. Clustering Fundamentals
- **Why taught**: Sets the stage. Students need to understand what we are trying to achieve before learning specific algorithms.
- **Nature**: Theory-heavy, visual explanation.
- **Focus**: Definition, types of clustering (partitional, hierarchical, density-based), distance metrics.

## 2. K-Means & Variants
- **Why taught**: The most common and simple clustering algorithm. 
- **Nature**: Code-heavy, business examples.
- **Focus**: The algorithm steps, the objective function (inertia), Elbow method, Silhouette score. K-Means++.

## 3. Hierarchical Clustering
- **Why taught**: Useful when the number of clusters is unknown or when a hierarchy/taxonomy is expected.
- **Nature**: Visual explanation (dendrograms).
- **Focus**: Agglomerative vs. Divisive, linkage criteria (Ward, Complete, Average), reading dendrograms.

## 4. Density-based Methods (DBSCAN)
- **Why taught**: K-Means fails on non-convex (weirdly shaped) clusters. DBSCAN handles these and finds outliers automatically.
- **Nature**: Theory and visual explanation.
- **Focus**: Core points, border points, noise points, `eps` and `min_samples` parameters.

## 5. Gaussian Mixture Models (GMM)
- **Why taught**: Introduces probabilistic clustering (soft clustering) and handles clusters with different variances.
- **Nature**: Theory-heavy, mathematical intuition.
- **Focus**: Expectation-Maximization (EM) algorithm, covariance types, AIC/BIC for model selection.
