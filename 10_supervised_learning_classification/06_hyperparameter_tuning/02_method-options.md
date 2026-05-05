# Method Options: Hyperparameter Tuning in Scikit-Learn

This document explains the search algorithms used for tuning.

## 1. `sklearn.model_selection.GridSearchCV`

### Syntax
```python
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_dict, cv=5, scoring='accuracy', n_jobs=-1)
```

### Common Arguments
- **`estimator`**: The model object you want to tune (e.g., `RandomForestClassifier()`).
- **`param_grid`** (`dict` or `list of dicts`): Dictionary with parameters names (string) as keys and lists of parameter settings to try as values.
  - Example: `{'max_depth': [3, 5, 10], 'C': [0.1, 1, 10]}`
- **`cv`** (`int`, default=`5`): Determines the cross-validation splitting strategy. Specifies the number of folds in a `(Stratified)KFold`.
- **`scoring`** (`str`, default=`None`): Strategy to evaluate the performance of the cross-validated model on the test set. For classification, options include `'accuracy'`, `'precision'`, `'recall'`, `'f1'`, `'roc_auc'`. If `None`, it uses the estimator's default `.score()` method.
- **`n_jobs`** (`int`, default=`None`): Number of jobs to run in parallel. `-1` means using all processors. Highly recommended.

### Useful Attributes After Fitting
- **`best_params_`**: The dictionary of parameters that gave the best results.
- **`best_estimator_`**: The actual model object, refitted on the *entire* training data using the best parameters. You can use this directly to `predict()`.
- **`cv_results_`**: A massive dictionary containing detailed logs of every combination tried and its score.

---

## 2. `sklearn.model_selection.RandomizedSearchCV`

### Syntax
```python
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats

param_dist = {'C': stats.uniform(0.1, 10), 'max_depth': [3, 5, 10]}
rand_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, cv=5)
```

### Common Arguments
- **`param_distributions`** (`dict`): Similar to `param_grid`, but values can be lists (discrete choices) or SciPy distributions (continuous ranges). 
- **`n_iter`** (`int`, default=`10`): Number of parameter settings that are sampled. Trades off runtime vs quality of the solution.

---

## 3. `sklearn.pipeline.Pipeline`

Crucial for advanced workflows. A pipeline chains multiple steps (like scaling and modeling) into one object.

### Syntax
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())
])
```

### Tuning Pipelines
When using `GridSearchCV` on a `Pipeline`, you must prefix the parameter names with the step name followed by a double underscore (`__`).
- Example: Instead of tuning `'C'`, you must tune `'svc__C'`.
