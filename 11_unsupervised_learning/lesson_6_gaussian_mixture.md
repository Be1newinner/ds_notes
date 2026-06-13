# Gaussian Mixture Models (GMM)

## Learning Objective
Understand probabilistic clustering using Gaussian Mixture Models and how the Expectation-Maximization algorithm works.

## What Is This Topic?
A Gaussian Mixture Model assumes that the data is generated from a mixture of a finite number of Gaussian distributions (bell curves) with unknown parameters.

## Why This Topic Matters
K-Means performs "hard clustering" (a point belongs strictly to one cluster). GMM performs "soft clustering" (a point has a probability of belonging to each cluster). Furthermore, GMM can handle elliptical clusters, not just spherical ones.

## Core Intuition
Imagine you have a dataset of heights. The data looks like a messy distribution. But underneath, it's actually two overlapping normal distributions: one for males, one for females. GMM tries to find those underlying distributions.

## Key Concepts
- Soft Clustering (Probabilities)
- Expectation-Maximization (EM) Algorithm
- Covariance Matrices (Shape of the cluster)

## Step-by-Step Explanation (EM Algorithm)
1. **Initialize:** Guess the mean, variance, and weight of K Gaussian distributions.
2. **Expectation (E-step):** For each data point, calculate the probability that it belongs to each distribution.
3. **Maximization (M-step):** Update the mean, variance, and weight of each distribution based on the probabilities calculated in the E-step.
4. **Repeat** E and M steps until the distributions stop changing (convergence).

## Important Parameters / Options / Settings
- `n_components`: The number of Gaussian distributions (K).
- `covariance_type`: Controls the shape of the clusters.
  - `spherical`: Circular, like K-Means.
  - `diag`: Elliptical, but aligned to axes.
  - `full`: Elliptical, can be rotated any way (most flexible, most prone to overfitting).

## Output / Result Interpretation
- `predict_proba`: Gives the probability matrix. Example: Point A has 90% chance of being Cluster 1, 10% chance of Cluster 2.
- `bic` or `aic`: Used to select the optimal number of components.

## Real-World Uses
- Speaker identification (voice data is highly probabilistic).
- Financial risk modeling.
- Generative AI (GMMs can generate new data points).

## Advantages
- Soft clustering provides a measure of uncertainty.
- Highly flexible cluster shapes.

## Limitations
- Slow to converge.
- Prone to finding local optima (bad initial guesses lead to bad results).
- If `covariance_type='full'`, it requires a lot of data and can easily overfit.

## Common Mistakes
- Using `full` covariance on small datasets.
- Not using AIC/BIC to find the optimal number of components.

## Related Methods
- K-Means (which is mathematically a special case of GMM where covariance is spherical and equal for all clusters).

## Code References
- `code/example-01-basic-gmm.py`
- `code/example-02-aic-bic.py`
- `code/example-03-real-world-generative.py`


---

## Gaussian Mixture Methods

### `sklearn.mixture.GaussianMixture`

#### Purpose
Fits a Gaussian mixture model.

#### Syntax
`model = GaussianMixture(n_components=3, covariance_type='full')`

#### Common Arguments
- `n_components` (int): Number of mixture components.
- `covariance_type` (str): 'full', 'tied', 'diag', 'spherical'.
- `init_params` (str): Method used to initialize weights ('kmeans' is default).

#### Common Attributes / Properties
- `means_`: Means of each mixture component.
- `covariances_`: Covariances of each mixture component.
- `weights_`: The mixing weights.

#### Output / Return Type
Fitted model object.

#### Typical Workflow
1. Scale data.
2. Fit GMMs with varying `n_components`.
3. Extract `model.bic(X)` to find the optimal number of components.
4. Fit final model.
5. Use `model.predict_proba(X)` to get soft assignments.

---

## Examples: Gaussian Mixture Models

### Code References
- `code/example-01-basic-gmm.py` — how GMM handles non-spherical, stretched data.
- `code/example-02-aic-bic.py` — using AIC/BIC to find the optimal number of components.
- `code/example-03-real-world-generative.py` — demonstrating soft probabilities and generative sampling.

---

## Practice: GMM
1. Run KMeans and GMM on the stretched data from Example 1. Plot both. Why does KMeans fail while GMM succeeds?
2. What is the difference between `predict` and `predict_proba` in a GMM?

---

## Interview Questions: Gaussian Mixture Models
1. What is "soft clustering" vs "hard clustering"?
2. Explain the Expectation-Maximization (EM) algorithm in simple terms.
3. How do AIC and BIC help in selecting the number of clusters?
4. When would you use a Gaussian Mixture Model instead of K-Means?

---

## Python Code Examples

### `example-01-basic-gmm.py`

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Generate stretched data
np.random.seed(42)
X, y_true = make_blobs(n_samples=400, centers=3, cluster_std=0.5, random_state=42)
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_stretched = np.dot(X, transformation)

# Fit GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
labels = gmm.fit_predict(X_stretched)

plt.scatter(X_stretched[:, 0], X_stretched[:, 1], c=labels, cmap='viridis')
plt.title("GMM Clustering (Handles Elliptical Shapes)")
plt.show()
```

### `example-02-aic-bic.py`

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)

n_components_range = range(1, 8)
bic_scores = []
aic_scores = []

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))
    aic_scores.append(gmm.aic(X))

plt.plot(n_components_range, bic_scores, marker='o', label='BIC')
plt.plot(n_components_range, aic_scores, marker='s', label='AIC')
plt.legend()
plt.title("Model Selection using AIC/BIC")
plt.xlabel("Number of Components")
plt.ylabel("Score (Lower is better)")
plt.show()
```

### `example-03-real-world-generative.py`

```python
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

# Historical Temperature and Ice Cream Sales
data = {
    'Temp_C': [25, 26, 28, 30, 32, 10, 12, 11, 15, 14],
    'Sales':  [200, 210, 250, 300, 310, 50, 60, 55, 80, 70]
}
df = pd.DataFrame(data)

gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(df)

# Soft Clustering Probabilities
probs = gmm.predict_proba(df)
df['Prob_Cluster_0'] = np.round(probs[:, 0], 2)
df['Prob_Cluster_1'] = np.round(probs[:, 1], 2)

print("Data with Probabilities:")
print(df)

# GMMs are generative! We can generate new synthetic data days
new_data, new_labels = gmm.sample(3)
print("\nGenerated Synthetic Days (Temp, Sales):")
print(np.round(new_data, 1))
```
