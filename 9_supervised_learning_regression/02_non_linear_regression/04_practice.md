# Practice: Non-Linear Regression

## Conceptual Questions
1. If you increase the `degree` of Polynomial Regression to 20, what will likely happen to your training error and your test error? What is this phenomenon called?
2. Why is the 'kernel trick' in SVR so important? 
3. How does the `epsilon` parameter in SVR act differently from the `alpha` penalty in Ridge/Lasso?

## Coding Tasks

### Task 1: Tuning Polynomial Degrees
1. Load a non-linear dataset (e.g., a subset of a housing dataset focusing on Age vs Price).
2. Create a loop that fits a `PolynomialFeatures` + `LinearRegression` pipeline for degrees 1, 2, 3, 5, and 10.
3. For each degree, calculate both the Training MSE and Test MSE.
4. Plot the MSEs on a graph (X-axis: Degree, Y-axis: Error). At what degree does the model start overfitting?

### Task 2: Understanding SVR Scaling
1. Use the `make_regression` function to generate data.
2. Train an `SVR(kernel='rbf')` model directly on the raw data and record the R2 score.
3. Apply `StandardScaler` to both X and y.
4. Train a new `SVR(kernel='rbf')` model on the scaled data.
5. Inverse transform your predictions and compare the R2 score with the unscaled model. Document the difference in performance.
