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


---

## Method & Options: Regression Metrics

This document details the common scikit-learn functions used to evaluate regression models. All these functions are found in `sklearn.metrics`.

### 1. `mean_absolute_error` (MAE)

#### Purpose
Calculates the average absolute distance between predicted and true values.

#### Syntax
```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
```
- **Output**: A float in the same units as your target variable (e.g., Dollars, Degrees). Lower is better.

---

### 2. `mean_squared_error` (MSE & RMSE)

#### Purpose
Calculates the average squared distance. It is used to get both MSE and RMSE.

#### Syntax for MSE
```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)
```

#### Syntax for RMSE
```python
from sklearn.metrics import mean_squared_error
# By setting squared=False, it automatically takes the square root
rmse = mean_squared_error(y_true, y_pred, squared=False) 
# Note: In newer versions of sklearn (1.4+), squared=False is deprecated. 
# Use root_mean_squared_error directly instead.
# from sklearn.metrics import root_mean_squared_error
# rmse = root_mean_squared_error(y_true, y_pred)
```
- **Output**: A float. RMSE is in the same units as the target variable. Lower is better.

---

### 3. `r2_score`

#### Purpose
Calculates the Coefficient of Determination ($R^2$). It measures how much better your model is than a "dumb" model that just predicts the average every time.

#### Syntax
```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
```
- **Output**: A float, usually between 0.0 and 1.0. 
- **Note**: It *can* be negative if your model is actually worse than just predicting the mean of the training data! Higher is better.

---

### 4. Adjusted R-squared

#### Purpose
Adjusts the $R^2$ score based on the number of features in your model. Scikit-learn does not have a built-in function for this, so you must calculate it manually using the standard $R^2$ score.

#### Syntax
```python
# n = number of samples (rows)
# p = number of features (columns)
n = len(y_test)
p = X_test.shape[1]

r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
```

#### Typical Workflow for Regression
When presenting a regression model, you should almost always print at least two metrics:
1. **RMSE or MAE**: To tell the business *how far off* the predictions will be in real dollars/units.
2. **R-squared**: To tell the data science team how well the model captured the variance in the data compared to the baseline.

---

## Regression Metrics Examples

This document explains the python examples provided in the `code/` directory.

### 1. MAE, MSE, and RMSE (`example-01-mae-mse-rmse.py`)
This script demonstrates the difference between the three primary error metrics.
- It generates a dataset with a few massive outliers (e.g., houses that sold for way more than they should have).
- It calculates MAE and RMSE.
- It highlights how RMSE results in a much larger number than MAE because RMSE heavily penalizes the model for missing those massive outliers.

### 2. R-squared and Adjusted R-squared (`example-02-r2-and-adjusted-r2.py`)
This script proves why standard R-squared can be dangerous.
- It creates a dataset with 5 useful features and calculates the R-squared.
- It then adds 50 completely random, useless features (noise) to the dataset.
- It shows how standard R-squared actually *increases* (or stays the same) with the useless features, tricking the user.
- It calculates Adjusted R-squared, which correctly decreases, warning the user that the new features are garbage.

### 3. Visualizing Residuals (`example-03-visualizing-residuals.py`)
Metrics are just numbers; visualizing the errors is crucial for debugging.
- It trains a model and calculates the residuals (`y_true - y_pred`).
- It plots the actual values vs. the predicted values.
- It plots a histogram of the residuals to show if the errors are normally distributed (a key assumption of Linear Regression).

---

## Practice Exercises: Regression Metrics

### Exercise 1: The Intuition of MAE vs RMSE
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

### Exercise 2: Implementing Metrics in Code
1. Load the `california_housing` dataset from `sklearn.datasets` (`fetch_california_housing`).
2. Do a train-test split (test_size=0.2, random_state=42).
3. Train a `LinearRegression` model.
4. Make predictions on the test set.
5. Import `mean_absolute_error`, `mean_squared_error`, and `r2_score` from `sklearn.metrics`.
6. Calculate and print all three metrics for your model.
7. Does the model have a "good" $R^2$ score? (Context: > 0.7 is generally considered okay).

### Exercise 3: Adjusted R-squared
1. Using the same code from Exercise 2, calculate the number of samples (`n`) in the test set.
2. Calculate the number of features (`p`) in the test set.
3. Write a python function `def adjusted_r2(y_test, y_pred, n, p):` that returns the Adjusted R-squared.
4. Does the Adjusted R-squared differ significantly from the standard R-squared in this specific dataset? Why or why not?

### Exercise 4: Visualizing the Errors
1. Calculate the residuals: `residuals = y_test - y_pred`.
2. Using `matplotlib.pyplot`, create a histogram of the residuals (`plt.hist(residuals, bins=50)`).
3. Do the residuals look like a normal bell curve centered at zero? If they are skewed to the right, what does that mean the model is doing wrong?

---

## Interview Questions: Regression Metrics

### Beginner Questions
1. **Can you use Accuracy to evaluate a regression model? Why or why not?**
   - *Answer*: No. Accuracy requires exact matches. In regression, predicting $300,000 for a house when the real price is $300,001 is technically "wrong" but practically perfect. Regression metrics must measure the *distance* (error) of the prediction, not just right or wrong.
2. **What does an R-squared of 0.85 mean?**
   - *Answer*: It means that 85% of the variance in the target variable can be explained by the features in the model. The remaining 15% is due to unobserved variables or random noise.

### Conceptual Questions
3. **What is the difference between MAE and RMSE?**
   - *Answer*: MAE is the simple average of absolute errors. RMSE squares the errors before averaging them, and then takes the square root. Because it squares the errors, RMSE heavily penalizes large outliers.
4. **If your boss asks you to explain how far off your model's predictions are, which metric do you use: MSE, RMSE, or R-squared?**
   - *Answer*: You should use RMSE or MAE. MSE is in "squared units" which makes no sense to humans. R-squared is a percentage of variance, not a dollar amount. RMSE or MAE gives the error in the original units (e.g., "We are off by $10,000").
5. **Why do we need Adjusted R-squared?**
   - *Answer*: Standard R-squared will mechanically increase (or stay the same) every time you add a new feature to the model, even if the feature is completely random noise. Adjusted R-squared penalizes the score for adding useless features, giving a more honest assessment of the model's true predictive power.

### Practical / Scenario Questions
6. **You are building an AI to predict when a patient will wake up from anesthesia. If the AI predicts early, the doctors just wait a bit longer. If the AI predicts late, the patient might wake up during surgery (a catastrophe). Which metric do you optimize?**
   - *Answer*: You must heavily penalize large errors, especially in one direction. You would likely use RMSE (or a custom asymmetric loss function) because a single massive error is unacceptable. MAE is too forgiving of outliers.
7. **You calculated an R-squared score of -0.15. Is this a bug in your code?**
   - *Answer*: Not necessarily a bug. An R-squared of 0.0 means your model is exactly as good as a "dumb" model that just predicts the average of the dataset every single time. A negative R-squared means your model is actually *worse* than just predicting the average. The model has failed to learn any useful pattern.

---

## Python Code Examples

### `example-01-mae-mse-rmse.py`

```python
"""
Example 01: MAE, MSE, and RMSE
This script highlights how RMSE punishes large outliers much more than MAE.
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 1. A scenario with small, normal errors
print("--- SCENARIO 1: Normal Errors ---")
y_true_1 = np.array([100, 150, 200, 250])
y_pred_1 = np.array([110, 140, 210, 240]) # Model is off by exactly 10 every time

mae_1 = mean_absolute_error(y_true_1, y_pred_1)
rmse_1 = mean_squared_error(y_true_1, y_pred_1, squared=False)

print(f"MAE:  {mae_1:.2f} (Average distance is 10)")
print(f"RMSE: {rmse_1:.2f} (Also 10, because all errors are exactly the same size)")
print()

# 2. A scenario with one MASSIVE outlier error
print("--- SCENARIO 2: One Massive Outlier ---")
y_true_2 = np.array([100, 150, 200, 250])
# Model is perfect for the first 3, but wildly wrong on the last one (off by 40)
y_pred_2 = np.array([100, 150, 200, 290]) 

mae_2 = mean_absolute_error(y_true_2, y_pred_2)
rmse_2 = mean_squared_error(y_true_2, y_pred_2, squared=False)

print(f"MAE:  {mae_2:.2f} (Total error of 40 / 4 houses = 10)")
print(f"RMSE: {rmse_2:.2f} (Much larger than MAE!)")

# Interpretation:
# Notice how both scenarios have an MAE of 10. The "total" amount of error is the same.
# But Scenario 2 has a much higher RMSE.
# This proves that RMSE heavily penalizes models that make single, massive mistakes.
```

### `example-02-r2-and-adjusted-r2.py`

```python
"""
Example 02: R-squared and Adjusted R-squared
This script shows how standard R-squared can be tricked by useless data.
"""

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# Helper function to calculate Adjusted R-squared
def adjusted_r2(r2, n_samples, n_features):
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

# 1. Base Model with 5 Good Features
X_good, y = make_regression(n_samples=200, n_features=5, noise=15, random_state=42)
X_train_g, X_test_g, y_train, y_test = train_test_split(X_good, y, test_size=0.2, random_state=42)

model_g = LinearRegression().fit(X_train_g, y_train)
y_pred_g = model_g.predict(X_test_g)

r2_g = r2_score(y_test, y_pred_g)
adj_r2_g = adjusted_r2(r2_g, len(y_test), X_test_g.shape[1])

print("--- MODEL WITH 5 USEFUL FEATURES ---")
print(f"R-squared:          {r2_g:.4f}")
print(f"Adjusted R-squared: {adj_r2_g:.4f}")
print()

# 2. Deceptive Model with 5 Good Features + 50 Useless Features (Random Noise)
X_noise = np.random.rand(200, 50)
X_bad = np.hstack([X_good, X_noise]) # Combine them
X_train_b, X_test_b, y_train, y_test = train_test_split(X_bad, y, test_size=0.2, random_state=42)

model_b = LinearRegression().fit(X_train_b, y_train)
y_pred_b = model_b.predict(X_test_b)

r2_b = r2_score(y_test, y_pred_b)
adj_r2_b = adjusted_r2(r2_b, len(y_test), X_test_b.shape[1])

print("--- MODEL WITH 50 ADDED USELESS FEATURES ---")
print(f"R-squared:          {r2_b:.4f}  <-- Notice how it barely changed, or even went up!")
print(f"Adjusted R-squared: {adj_r2_b:.4f}  <-- Notice how it crashed. It knows the new features are garbage.")
```

### `example-03-visualizing-residuals.py`

```python
"""
Example 03: Visualizing Residuals
Metrics are just numbers. You must visualize your errors to truly understand them.
"""

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Train a model
housing = fetch_california_housing()
X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# 2. Calculate Residuals (Errors)
residuals = y_test - y_pred

# 3. Create Visualizations
plt.figure(figsize=(12, 5))

# Plot 1: Actual vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.3)
# Draw the "Perfect Prediction" diagonal line
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted")

# Plot 2: Residual Histogram
plt.subplot(1, 2, 2)
plt.hist(residuals, bins=50, edgecolor='black')
plt.axvline(x=0, color='r', linestyle='--')
plt.xlabel("Prediction Error (Residuals)")
plt.ylabel("Count")
plt.title("Histogram of Residuals")

plt.tight_layout()
plt.savefig("residuals_plot.png")
print("Saved visualization to residuals_plot.png")

# Interpretation:
# The histogram should look like a normal bell curve centered around the red line (Zero error).
# If it is wildly skewed, your model is systematically over-predicting or under-predicting.
```
