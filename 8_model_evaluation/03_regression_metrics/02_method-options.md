# Method & Options: Regression Metrics

This document details the common scikit-learn functions used to evaluate regression models. All these functions are found in `sklearn.metrics`.

## 1. `mean_absolute_error` (MAE)

### Purpose
Calculates the average absolute distance between predicted and true values.

### Syntax
```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
```
- **Output**: A float in the same units as your target variable (e.g., Dollars, Degrees). Lower is better.

---

## 2. `mean_squared_error` (MSE & RMSE)

### Purpose
Calculates the average squared distance. It is used to get both MSE and RMSE.

### Syntax for MSE
```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)
```

### Syntax for RMSE
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

## 3. `r2_score`

### Purpose
Calculates the Coefficient of Determination ($R^2$). It measures how much better your model is than a "dumb" model that just predicts the average every time.

### Syntax
```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
```
- **Output**: A float, usually between 0.0 and 1.0. 
- **Note**: It *can* be negative if your model is actually worse than just predicting the mean of the training data! Higher is better.

---

## 4. Adjusted R-squared

### Purpose
Adjusts the $R^2$ score based on the number of features in your model. Scikit-learn does not have a built-in function for this, so you must calculate it manually using the standard $R^2$ score.

### Syntax
```python
# n = number of samples (rows)
# p = number of features (columns)
n = len(y_test)
p = X_test.shape[1]

r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
```

### Typical Workflow for Regression
When presenting a regression model, you should almost always print at least two metrics:
1. **RMSE or MAE**: To tell the business *how far off* the predictions will be in real dollars/units.
2. **R-squared**: To tell the data science team how well the model captured the variance in the data compared to the baseline.
