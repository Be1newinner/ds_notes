# Linear Regression and Regularization

## Learning Objective
Students will understand the core mechanism of fitting a straight line to data, handling multiple variables, and applying regularization (Ridge, Lasso) to prevent overfitting.

## What Is This Topic?
Linear Regression predicts a continuous numerical output by drawing a "line of best fit" through the data points. Regularization is an advanced form of linear regression that adds a penalty to the model to keep it simple and prevent it from memorizing the noise in the data.

## Why This Topic Matters
It is the most fundamental algorithm in machine learning. It provides a baseline for predictive modeling, is highly interpretable, and is widely used in finance, economics, and business forecasting.

## Core Intuition
Imagine you have data points scattered on a graph showing house size vs. house price. Linear regression draws a straight line through the middle of these points so that the total distance from the points to the line is as small as possible. Regularization (like putting a speed limit on a car) prevents the model's coefficients (the slope of the line) from becoming too extreme.

## Key Concepts
- **Weights (Coefficients)**: How much each feature influences the prediction.
- **Bias (Intercept)**: The starting baseline prediction if all features were zero.
- **Cost Function (MSE)**: The metric the model tries to minimize (the error).
- **Gradient Descent**: The optimizer that slowly adjusts the line to find the best fit.
- **L1 Regularization (Lasso)**: Can shrink some weights exactly to zero, doing automatic feature selection.
- **L2 Regularization (Ridge)**: Shrinks weights to be small but rarely exactly zero.

## Step-by-Step Explanation
1. Initialize the line with random weights and a bias.
2. Make predictions for all data points.
3. Calculate the Mean Squared Error (MSE) between predictions and actual values.
4. Use an optimizer to adjust the weights and bias slightly to lower the error.
5. Repeat until the error stops decreasing (convergence).

## Important Parameters / Options / Settings
- `fit_intercept`: Should the model calculate the baseline (intercept)? Almost always `True`.
- `alpha` (in Ridge/Lasso): The strength of the regularization penalty. Higher `alpha` means a simpler model.

## Output / Result Interpretation
The output is a continuous numerical value (e.g., predicted price = $250,000). The coefficients tell you the relationship: a coefficient of 50 for 'size' means for every 1 sq ft increase, the price goes up by $50.

## Real-World Uses
- Predicting next quarter's revenue based on marketing spend across channels.
- Estimating the yield of a crop based on rainfall and fertilizer amount.
- Determining the price elasticity of a product.

## Advantages
- Fast to train and predict.
- Highly interpretable (you can explain exactly how a prediction was made).
- Works very well when the relationship is genuinely linear.

## Limitations
- Cannot capture complex, non-linear relationships without feature engineering.
- Very sensitive to outliers.
- Assumes features are independent (multicollinearity can cause issues).

## Common Mistakes
- Not scaling data before using Ridge or Lasso.
- Ignoring outliers that pull the line of best fit away from the majority of data.
- Applying linear regression when the relationship is clearly a curve.

## Related Methods
- Polynomial Regression (adds non-linear terms).
- Logistic Regression (used for classification, not regression).

## Code References
- `code/example-01-basic-simple-linear.py` — simple introduction example
- `code/example-02-intermediate-multiple-linear.py` — handling multiple features
- `code/example-03-real-world-regularization.py` — applying Ridge/Lasso to real data
