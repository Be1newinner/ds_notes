# Practice Exercises: Regression Metrics

## Exercise 1: The Intuition of MAE vs RMSE
Imagine a dataset with 5 houses. Your model predicts the prices.
The True Prices are: `[100k, 150k, 200k, 250k, 300k]`
Your Predictions are: `[110k, 140k, 200k, 260k, 400k]`

**Task:**
1. Calculate the Error (Residual) for each house.
2. Calculate the Absolute Error for each house.
3. Calculate the MAE (Mean Absolute Error) manually.
4. Calculate the Squared Error for each house.
5. Calculate the MSE manually.
6. Calculate the RMSE manually.
7. Notice how much larger the RMSE is than the MAE. Which house caused this massive difference?

## Exercise 2: Implementing Metrics in Code
1. Load the `california_housing` dataset from `sklearn.datasets` (`fetch_california_housing`).
2. Do a train-test split (test_size=0.2, random_state=42).
3. Train a `LinearRegression` model.
4. Make predictions on the test set.
5. Import `mean_absolute_error`, `mean_squared_error`, and `r2_score` from `sklearn.metrics`.
6. Calculate and print all three metrics for your model.
7. Does the model have a "good" $R^2$ score? (Context: > 0.7 is generally considered okay).

## Exercise 3: Adjusted R-squared
1. Using the same code from Exercise 2, calculate the number of samples (`n`) in the test set.
2. Calculate the number of features (`p`) in the test set.
3. Write a python function `def adjusted_r2(y_test, y_pred, n, p):` that returns the Adjusted R-squared.
4. Does the Adjusted R-squared differ significantly from the standard R-squared in this specific dataset? Why or why not?

## Exercise 4: Visualizing the Errors
1. Calculate the residuals: `residuals = y_test - y_pred`.
2. Using `matplotlib.pyplot`, create a histogram of the residuals (`plt.hist(residuals, bins=50)`).
3. Do the residuals look like a normal bell curve centered at zero? If they are skewed to the right, what does that mean the model is doing wrong?
