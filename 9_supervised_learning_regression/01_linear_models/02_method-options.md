# Method Options: Linear Models in Scikit-Learn

This file covers the key classes and methods used for Linear Regression and Regularization in Scikit-Learn.

## `sklearn.linear_model.LinearRegression`

**Purpose**: Performs standard Ordinary Least Squares (OLS) Linear Regression.

**Syntax**:
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
```

**Common Arguments**:
- `fit_intercept` (bool, default `True`): Whether to calculate the intercept. Set to `False` if data is already centered around the origin.
- `n_jobs` (int, default `None`): Number of jobs to use for computation. `-1` uses all processors (useful for huge datasets or multiple targets).

**Common Attributes**:
- `coef_`: Array of shape (n_features,). The estimated coefficients for the features.
- `intercept_`: Float. The independent term in the linear model.

**Typical Workflow**:
```python
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model.coef_, model.intercept_)
```

---

## `sklearn.linear_model.Ridge`

**Purpose**: Linear regression with L2 regularization. Useful for handling multicollinearity and preventing overfitting.

**Syntax**:
```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
```

**Common Arguments**:
- `alpha` (float, default `1.0`): Regularization strength. Larger values specify stronger regularization.
- `solver` (str, default `'auto'`): Algorithm to use in the computational routines.
- `random_state` (int): Seed for reproducibility when using stochastic solvers.

**Common Attributes**:
- `coef_`, `intercept_` (same as LinearRegression).

**Important Note**: Data **must** be scaled (e.g., using `StandardScaler`) before applying Ridge, as the penalty depends on the scale of the features.

---

## `sklearn.linear_model.Lasso`

**Purpose**: Linear regression with L1 regularization. Useful for feature selection because it can shrink coefficients exactly to zero.

**Syntax**:
```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)
```

**Common Arguments**:
- `alpha` (float, default `1.0`): Regularization strength. Higher values push more coefficients to zero.
- `max_iter` (int, default `1000`): Maximum number of iterations for the solver.

---

## Output / Return Type
The `predict(X)` method returns a 1D numpy array of shape (n_samples,) containing the continuous numerical predictions.

## Common Mistakes
- **Forgetting to Scale Data**: Applying Ridge or Lasso without standardizing features causes features with larger scales to be penalized unfairly.
- **Misinterpreting Coefficients without Scaling**: You cannot compare the importance of `coef_` values if the features have different units (e.g., Age in years vs. Salary in dollars) unless the data was scaled first.
