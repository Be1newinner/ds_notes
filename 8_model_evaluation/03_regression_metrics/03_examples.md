# Regression Metrics Examples

This document explains the python examples provided in the `code/` directory.

## 1. MAE, MSE, and RMSE (`example-01-mae-mse-rmse.py`)
This script demonstrates the difference between the three primary error metrics.
- It generates a dataset with a few massive outliers (e.g., houses that sold for way more than they should have).
- It calculates MAE and RMSE.
- It highlights how RMSE results in a much larger number than MAE because RMSE heavily penalizes the model for missing those massive outliers.

## 2. R-squared and Adjusted R-squared (`example-02-r2-and-adjusted-r2.py`)
This script proves why standard R-squared can be dangerous.
- It creates a dataset with 5 useful features and calculates the R-squared.
- It then adds 50 completely random, useless features (noise) to the dataset.
- It shows how standard R-squared actually *increases* (or stays the same) with the useless features, tricking the user.
- It calculates Adjusted R-squared, which correctly decreases, warning the user that the new features are garbage.

## 3. Visualizing Residuals (`example-03-visualizing-residuals.py`)
Metrics are just numbers; visualizing the errors is crucial for debugging.
- It trains a model and calculates the residuals (`y_true - y_pred`).
- It plots the actual values vs. the predicted values.
- It plots a histogram of the residuals to show if the errors are normally distributed (a key assumption of Linear Regression).
