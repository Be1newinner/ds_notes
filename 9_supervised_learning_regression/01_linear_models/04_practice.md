# Practice: Linear Models

## Conceptual Questions
1. If you fit a Linear Regression model and the Mean Squared Error is very high on both training and test data, is the model suffering from high bias or high variance?
2. Why is it dangerous to use a standard Linear Regression model on a dataset that has severe outliers?
3. What is the difference between Lasso and Ridge regression in terms of how they handle the feature coefficients?
4. If you have 10,000 features but only 100 of them are actually useful for prediction, which regularization technique (Lasso or Ridge) would be better and why?

## Coding Tasks

### Task 1: Simple Linear Regression
1. Generate synthetic data using `sklearn.datasets.make_regression` with `n_features=1` and `noise=20`.
2. Split the data into train and test sets.
3. Train a `LinearRegression` model.
4. Predict on the test set and calculate the Mean Absolute Error (MAE).
5. Plot the original data points (scatter plot) and the regression line (line plot) using Matplotlib.

### Task 2: Impact of Scaling on Regularization
1. Load a dataset with differently scaled features (e.g., California Housing dataset).
2. Train a `Ridge(alpha=10)` model on the raw, unscaled training data and record the test R2 score.
3. Apply `StandardScaler` to the training data, scale the test data, and train a new `Ridge(alpha=10)` model.
4. Compare the R2 scores and the model coefficients before and after scaling. What do you observe?

### Task 3: Feature Selection with Lasso
1. Create a dataset with 50 features using `make_regression` (set `n_informative=5`).
2. Scale the data.
3. Train a `Lasso` regression model with an appropriate `alpha` (e.g., 0.5).
4. Print out the model coefficients. How many coefficients were shrunk exactly to zero?
