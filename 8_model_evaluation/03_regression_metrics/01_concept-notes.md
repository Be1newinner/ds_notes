# Regression Metrics

## Learning Objective
Students will learn how to evaluate models that predict continuous numbers (e.g., price, temperature, age). They will understand how to measure the size of a model's errors and how to explain the model's accuracy to business stakeholders.

## What Is This Topic?
In regression, we are not predicting "Yes" or "No". We are predicting "How much?" Therefore, there is no such thing as being 100% "accurate". A model that predicts a house price as $300,000 when the true price is $300,001 is functionally perfect, even though it was technically wrong. Regression metrics measure the *distance* between the predictions and the actual values (the "Error" or "Residual").

## Why This Topic Matters
Different business problems require different ways of penalizing errors. If you are predicting the temperature for a picnic, being off by 10 degrees is bad, but not catastrophic. If you are a machine learning trading bot, being off by 10 dollars on a stock price could bankrupt the company. We need specific mathematical formulas to treat errors differently based on the context.

## Core Intuition
Imagine throwing darts at a dartboard.
- The bullseye is the actual value (`y_true`).
- Where your dart lands is the prediction (`y_pred`).
- The distance between the dart and the bullseye is the **Error** (or Residual).
Regression metrics are just different ways of calculating the average distance of all your darts.

## Key Concepts

### 1. MAE (Mean Absolute Error)
The average of the absolute differences between predictions and actual values.
- **Interpretation**: "On average, our predictions are off by X amount."
- **Pros**: Very easy for business stakeholders to understand. Does not severely punish extreme outliers.

### 2. MSE (Mean Squared Error)
The average of the *squared* differences between predictions and actual values.
- **Interpretation**: Hard to interpret directly because the units are squared (e.g., "squared dollars").
- **Pros**: Mathematically convenient for optimization algorithms (calculus). Severely punishes large errors.

### 3. RMSE (Root Mean Squared Error)
The square root of the MSE.
- **Interpretation**: Back in the original units (e.g., dollars). Like MAE, but heavily penalizes large errors.
- **When to use**: When a single massive error is much worse than many small errors.

### 4. R-squared ($R^2$)
The proportion of the variance in the target variable that is predictable from the features.
- **Interpretation**: A score between 0 and 1. An $R^2$ of 0.80 means "Our model explains 80% of the reasons why the price fluctuates."
- **Pros**: It is unitless. You can use it to compare a model predicting House Prices (in millions) against a model predicting Temperature (in degrees).

### 5. Adjusted R-squared
Standard $R^2$ artificially goes up every time you add a new feature, even if the feature is garbage (like adding "Phase of the Moon" to predict house prices). Adjusted $R^2$ penalizes you for adding useless features.

## Real-World Uses
- **Predicting Delivery Times**: You might optimize for MAE. If the pizza is 5 minutes late or 10 minutes late, the customer is slightly annoyed either way.
- **Self-Driving Car Braking Distance**: You MUST optimize for RMSE. Being off by 1 inch 100 times is fine. Being off by 100 inches exactly 1 time means a crash. Large errors must be heavily penalized.

## Common Mistakes
- **Using Classification metrics for Regression**: You cannot calculate Accuracy, Precision, or Recall on a continuous variable.
- **Relying solely on $R^2$**: A model can have a high $R^2$ but still have an MAE that is too large for the business to actually use. Always report an error metric (MAE/RMSE) alongside $R^2$.

## Related Methods
- **Residual Plots**: Visualizing the errors on a scatter plot to ensure the model isn't systematically over-predicting or under-predicting.

## Code References
- `code/example-01-mae-mse-rmse.py`
- `code/example-02-r2-and-adjusted-r2.py`
- `code/example-03-visualizing-residuals.py`
