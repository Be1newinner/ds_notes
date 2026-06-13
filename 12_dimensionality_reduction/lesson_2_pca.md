# Principal Component Analysis (PCA)

## Learning Objective
Understand how to reduce the number of features in a dataset while retaining the maximum amount of variance (information), and how to apply PCA using scikit-learn.

## What Is This Topic?
PCA is a technique that transforms high-dimensional data into a lower-dimensional form by finding new axes (Principal Components) that maximize the variance of the data.

## Why This Topic Matters
When you have hundreds of features, models become slow and prone to overfitting (curse of dimensionality). PCA compresses these features into a smaller set of uncorrelated features, speeding up training and helping with data visualization.

## Core Intuition
Imagine looking at a 3D teapot. Depending on the angle you view it from, you might see just a circle (top view) or the full shape (side view). PCA mathematically finds the "best angle" (the axis) to look at the data so that the points are most spread out (highest variance), ensuring you see the most important shape of the data.

## Key Concepts
- **Variance**: How spread out the data is. PCA tries to maximize this.
- **Principal Components (PCs)**: New, artificial features created by PCA. They are linear combinations of the original features. PC1 captures the most variance, PC2 captures the second most, and so on.
- **Orthogonality**: Principal components are perpendicular to each other, meaning they are completely uncorrelated.
- **Explained Variance**: The percentage of total information captured by each principal component.

## Step-by-Step Explanation
1. Standardize the data (mean=0, variance=1) because PCA is highly sensitive to the scale of the features.
2. Compute the Covariance Matrix to understand how features relate to each other.
3. Calculate Eigenvalues and Eigenvectors of the covariance matrix.
4. Sort Eigenvectors by their Eigenvalues in descending order (these are your PCs).
5. Project the original data onto these new axes.

## Important Parameters / Options / Settings
- `n_components`: Number of components to keep. Can be an integer (e.g., 2 for 2D plot) or a float between 0 and 1 (e.g., 0.95 to keep 95% of the variance).
- `random_state`: Used for certain solvers to ensure reproducible results.

## Output / Result Interpretation
If `pca.explained_variance_ratio_` for a 2-component PCA is `[0.60, 0.25]`, it means PC1 captures 60% of the original data's information, and PC2 captures 25%. Together, a 2D scatter plot will represent 85% of the original dataset's complexity.

## Real-World Uses
- Feature extraction before feeding data into an SVM or Neural Network.
- Visualizing a 50-feature dataset in a 2D scatter plot to see if natural clusters exist.
- Image compression.

## Advantages
- Removes multicollinearity (PCs are independent).
- Reduces training time and storage space.
- Makes visualization of high-dimensional data possible.

## Limitations
- **Loss of Interpretability**: Principal components are combinations of original features (e.g., PC1 = 0.5*Age + 0.3*Income), making them hard to explain to business stakeholders.
- **Assumes Linear Relationships**: Fails to capture non-linear patterns.
- **Scale Sensitive**: Requires standard scaling before applying.

## Common Mistakes
- Applying PCA without standardizing the data first (features with larger scales will dominate the components).
- Using PCA on categorical variables (PCA is designed for continuous numerical data).
- Trying to interpret the new features as direct physical quantities.

## Related Methods
- t-SNE / UMAP (for non-linear reduction and visualization).
- SVD (Singular Value Decomposition - the underlying math for most PCA implementations).
- LDA (Linear Discriminant Analysis - supervised dimensionality reduction).

## Code References
- `code/example-01-basic.py`
- `code/example-02-intermediate.py`
- `code/example-03-real-world.py`


---

## PCA Method and Options

### Scikit-Learn: `sklearn.decomposition.PCA`

#### Purpose
To perform linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.

#### Syntax
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
```

#### Common Arguments
- `n_components` (int, float, or None): Number of components to keep.
  - If `int` (e.g., `2`): Keeps exactly 2 components.
  - If `float` (e.g., `0.95`): Keeps enough components to explain 95% of the variance.
  - If `None`: Keeps all components (useful for looking at the cumulative variance plot).
- `whiten` (bool): When True, it scales the components to have unit variance. Can sometimes improve predictive accuracy of downstream models.
- `random_state` (int): Seed for reproducibility when using randomized solvers.

#### Common Methods
- `fit(X)`: Learns the principal components from the training data X.
- `transform(X)`: Applies the dimensionality reduction to X, returning the new, lower-dimensional data.
- `fit_transform(X)`: Fits the model and transforms the data in one step (more efficient for training data).
- `inverse_transform(X_transformed)`: Transforms data back to its original space (useful for checking compression loss).

#### Common Attributes
- `components_`: The principal axes in feature space, representing the directions of maximum variance (the eigenvectors). Shape is `(n_components, n_features)`.
- `explained_variance_`: The amount of variance explained by each of the selected components (the eigenvalues).
- `explained_variance_ratio_`: Percentage of variance explained by each component. Very useful for plotting the "Scree Plot".

#### Typical Workflow
1. Import `StandardScaler` and `PCA`.
2. Fit and transform the data with `StandardScaler`.
3. Initialize `PCA` with desired components.
4. Call `fit_transform()` on the scaled data.
5. Check `explained_variance_ratio_.sum()` to see total information retained.

#### Common Mistakes
- **Forgetting `StandardScaler`**: This is the #1 mistake. PCA maximizes variance, so a feature measured in millions will dominate a feature measured in decimals if not scaled.
- **Fitting PCA on Test Data**: Always `fit` PCA on the training set, and only `transform` the test set to avoid data leakage.

---

## PCA Code Examples Overview

Here are the code examples provided in the `code/` folder to demonstrate PCA:

### 1. `code/example-01-basic.py`
A simple introduction to PCA using synthetic data. It shows how to import PCA, fit it, and transform a 3D dataset into a 2D dataset. It also demonstrates how to check the `explained_variance_ratio_`.

### 2. `code/example-02-intermediate.py`
Demonstrates the importance of feature scaling (StandardScaler) before applying PCA, and shows how to plot a Scree Plot (Cumulative Explained Variance) to decide how many components to keep.

### 3. `code/example-03-real-world.py`
A practical example using a real-world dataset (like Breast Cancer dataset). It shows how to compress the features, plot the 2D results to see class separation, and how a classifier performs on the reduced dataset compared to the original.

---

## PCA Practice Tasks

### Task 1: Scaling Check
Generate a random dataset with 2 features: one ranging from 0 to 1, and another from 0 to 1,000,000. Apply PCA without scaling and check the `components_`. Then apply `StandardScaler` and apply PCA again. Compare the `components_` and explain the difference.

### Task 2: 95% Variance
Load the `load_digits` dataset from `sklearn.datasets`.
1. Scale the data.
2. Fit PCA without specifying `n_components`.
3. Find out exactly how many components are needed to explain 95% of the variance.
4. Re-run PCA with that exact number of components.

### Task 3: Visualization
Load the `wine` dataset from `sklearn.datasets`.
1. Scale the data.
2. Reduce it to 2 components using PCA.
3. Plot the result using a scatter plot, coloring the points by the `target` variable.

---

## PCA Interview Questions

1. **Beginner**: What is the main goal of PCA?
2. **Conceptual**: Why must you standardize data before applying PCA? What happens if you don't?
3. **Conceptual**: Can you explain PCA to a non-technical manager?
4. **Practical**: How do you decide how many principal components to keep?
5. **Practical**: Are the new features created by PCA correlated with each other? Why or why not?
6. **Comparison**: What is the difference between Feature Selection and Feature Extraction (like PCA)?
7. **Output**: If the `explained_variance_ratio_` for 3 components is `[0.40, 0.30, 0.05]`, what does this mean? How much total variance is captured?
8. **Advanced**: How does PCA handle non-linear data? (Hint: It doesn't handle it well, which is why Kernel PCA or t-SNE is used).

---

## Python Code Examples

### `example-01-basic.py`

```python
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# 1. Create a simple synthetic dataset (3 dimensions)
np.random.seed(42)
data = {
    'length': np.random.normal(10, 2, 100),
    'width': np.random.normal(5, 1, 100),
    'height': np.random.normal(15, 3, 100)
}
df = pd.DataFrame(data)

print("Original Data Shape:", df.shape)
print(df.head(3))
print("-" * 30)

# 2. Initialize PCA to reduce to 2 dimensions
pca = PCA(n_components=2)

# 3. Fit and transform the data
transformed_data = pca.fit_transform(df)

print("Transformed Data Shape:", transformed_data.shape)
print(transformed_data[:3])
print("-" * 30)

# 4. Check how much information we retained
variance_ratio = pca.explained_variance_ratio_
print(f"Variance captured by PC1: {variance_ratio[0]:.2%}")
print(f"Variance captured by PC2: {variance_ratio[1]:.2%}")
print(f"Total variance captured: {np.sum(variance_ratio):.2%}")
```

### `example-02-intermediate.py`

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Generate data with 20 features
X, y = make_classification(n_samples=500, n_features=20, n_informative=5, random_state=42)

# 2. MANDATORY: Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Fit PCA without limiting components to see the variance spread
pca_full = PCA()
pca_full.fit(X_scaled)

# 4. Calculate cumulative variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# 5. Plot the Scree Plot (Cumulative Variance)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.axhline(y=0.90, color='r', linestyle='-', label='90% Variance Threshold')
plt.title('Explained Variance by Number of Principal Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend()
plt.grid(True)
plt.show()

# 6. Apply PCA keeping 90% of the variance
pca_90 = PCA(n_components=0.90)
X_reduced = pca_90.fit_transform(X_scaled)
print(f"Original features: {X.shape[1]}")
print(f"Reduced features (for 90% variance): {X_reduced.shape[1]}")
```

### `example-03-real-world.py`

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load Real World Dataset (Breast Cancer: 30 features)
data = load_breast_cancer()
X = data.data
y = data.target
target_names = data.target_names

# 2. Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply PCA to reduce 30 features to 2 features for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_pca.shape}")
print(f"Total variance explained by 2 components: {pca.explained_variance_ratio_.sum():.2%}")

# 4. Plot the results in 2D
plt.figure(figsize=(8, 6))
colors = ['red', 'green']

for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=0.7, lw=2, label=target_name)

plt.title('PCA of Breast Cancer Dataset (2 Components)')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.grid(True)
plt.show()
```
