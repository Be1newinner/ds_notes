# PCA Code Examples Overview

Here are the code examples provided in the `code/` folder to demonstrate PCA:

## 1. `code/example-01-basic.py`
A simple introduction to PCA using synthetic data. It shows how to import PCA, fit it, and transform a 3D dataset into a 2D dataset. It also demonstrates how to check the `explained_variance_ratio_`.

## 2. `code/example-02-intermediate.py`
Demonstrates the importance of feature scaling (StandardScaler) before applying PCA, and shows how to plot a Scree Plot (Cumulative Explained Variance) to decide how many components to keep.

## 3. `code/example-03-real-world.py`
A practical example using a real-world dataset (like Breast Cancer dataset). It shows how to compress the features, plot the 2D results to see class separation, and how a classifier performs on the reduced dataset compared to the original.
