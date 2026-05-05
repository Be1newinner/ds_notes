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
