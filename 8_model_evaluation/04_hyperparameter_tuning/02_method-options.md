# Method & Options: Hyperparameter Tuning

This document details the common scikit-learn functions used to tune models. They are found in `sklearn.model_selection`.

## 1. `GridSearchCV`

### Purpose
Performs an exhaustive search over a specified parameter grid, using cross-validation to evaluate each combination.

### Syntax
```python
from sklearn.model_selection import GridSearchCV

# Define the dictionary of settings to try
param_grid = {
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5]
}

# Initialize the searcher
grid_search = GridSearchCV(
    estimator=model, 
    param_grid=param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1 
)

# Run the search (This takes time!)
grid_search.fit(X_train, y_train)
```

### Common Arguments
- `estimator`: The untrained machine learning model.
- `param_grid` (dict): Dictionary with parameters names (string) as keys and lists of parameter settings to try as values.
- `cv` (int): Number of Cross-Validation folds. (e.g., `5` means every combination is trained 5 times).
- `scoring` (string): The metric to optimize for (e.g., `'f1'`, `'neg_mean_squared_error'`).
- `n_jobs` (int): Number of CPU cores to use. Set to `-1` to use all available cores (highly recommended to speed it up).

### Important Attributes (After fitting)
- `grid_search.best_params_`: A dictionary showing the winning combination of settings.
- `grid_search.best_estimator_`: The actual trained model using the winning settings. You can immediately call `.predict()` on this.
- `grid_search.best_score_`: The cross-validation score the winning model achieved.

---

## 2. `RandomizedSearchCV`

### Purpose
Performs a randomized search over hyperparameters. Instead of trying every combination, it samples randomly from distributions.

### Syntax
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Define distributions instead of lists
param_dist = {
    'max_depth': randint(3, 20),      # Random integers between 3 and 20
    'min_samples_split': randint(2, 10)
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=20,     # Try exactly 20 random combinations
    cv=5,
    random_state=42
)

random_search.fit(X_train, y_train)
```

### Common Arguments
- `param_distributions` (dict): Dictionary with parameter names as keys and distributions (like `scipy.stats.randint`) or lists as values.
- `n_iter` (int): Number of parameter settings that are sampled. Trades off runtime vs quality of the solution.

### Workflow Best Practice
1. Use `RandomizedSearchCV` first with a wide range of values and `n_iter=50` to find roughly where the best parameters live.
2. Look at the `best_params_`.
3. Create a tight `param_grid` around those specific values and run `GridSearchCV` to find the exact mathematical peak.
