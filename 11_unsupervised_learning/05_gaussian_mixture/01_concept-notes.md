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
