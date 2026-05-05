# Method Options: Evaluation and Tuning

## Regression Metrics in `sklearn.metrics`

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# MAE
mae = mean_absolute_error(y_true, y_pred)

# MSE
mse = mean_squared_error(y_true, y_pred)

# RMSE (pass squared=False to mean_squared_error)
rmse = mean_squared_error(y_true, y_pred, squared=False)
# OR manually
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# R-squared
r2 = r2_score(y_true, y_pred)
```

---

## `sklearn.model_selection.GridSearchCV`

**Purpose**: Exhaustive search over specified parameter values for an estimator. Includes built-in Cross-Validation.

**Syntax**:
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20]
}

grid_search = GridSearchCV(
    estimator=model, 
    param_grid=param_grid, 
    cv=5, 
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
```

**Key Arguments**:
- `estimator`: The model you want to tune.
- `param_grid` (dict): Dictionary with parameters names as keys and lists of parameter settings to try as values.
- `cv` (int, default `5`): Number of folds for cross-validation.
- `scoring` (str): The metric to evaluate. Note: `GridSearchCV` always tries to *maximize* the score. Therefore, loss metrics like MSE are passed as `'neg_mean_squared_error'` (negative MSE) so that maximizing it means minimizing the actual error.
- `n_jobs` (int, default `None`): Set to `-1` to use all CPU cores.

**Common Attributes**:
- `best_params_`: Dict of the parameters that gave the best results.
- `best_estimator_`: The actual model instance with the best parameters, already retrained on the whole dataset.

---

## `sklearn.model_selection.RandomizedSearchCV`

**Purpose**: Randomized search on hyper parameters. Faster alternative to GridSearchCV when the parameter space is huge.

**Syntax**:
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': [None, 10, 20, 30]
}

random_search = RandomizedSearchCV(
    estimator=model, 
    param_distributions=param_dist, 
    n_iter=10,  # Number of random combinations to try
    cv=5, 
    scoring='neg_mean_squared_error',
    random_state=42
)
```

**Key Difference**: You use `param_distributions` instead of `param_grid`. You can pass lists, or probability distributions (like `scipy.stats.randint` or `uniform`) and it will randomly sample `n_iter` combinations from them.
