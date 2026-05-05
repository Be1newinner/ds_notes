# PCA Practice Tasks

## Task 1: Scaling Check
Generate a random dataset with 2 features: one ranging from 0 to 1, and another from 0 to 1,000,000. Apply PCA without scaling and check the `components_`. Then apply `StandardScaler` and apply PCA again. Compare the `components_` and explain the difference.

## Task 2: 95% Variance
Load the `load_digits` dataset from `sklearn.datasets`.
1. Scale the data.
2. Fit PCA without specifying `n_components`.
3. Find out exactly how many components are needed to explain 95% of the variance.
4. Re-run PCA with that exact number of components.

## Task 3: Visualization
Load the `wine` dataset from `sklearn.datasets`.
1. Scale the data.
2. Reduce it to 2 components using PCA.
3. Plot the result using a scatter plot, coloring the points by the `target` variable.
