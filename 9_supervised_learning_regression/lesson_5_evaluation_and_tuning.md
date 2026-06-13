# Evaluation and Tuning for Regression Models

## Learning Objective
Students will learn how to quantify the errors made by regression models, translate those mathematical errors into business logic, and use hyperparameter tuning to optimize model performance.

## What Is This Topic?
Evaluation metrics tell us *how wrong* our model's predictions are. Tuning is the process of adjusting the "knobs and dials" (hyperparameters) of an algorithm to make those errors as small as possible without overfitting.

## Why This Topic Matters
An R-squared of 0.90 sounds great, but if you are predicting heart rates, an error of 10 beats per minute might be fatal. You must know how to translate statistical metrics into real-world impact. Tuning separates a decent model from a production-ready model.

## Core Evaluation Metrics
- **Mean Absolute Error (MAE)**: The average of the absolute differences between predictions and actual values. Highly interpretable (e.g., "We are off by $500 on average").
- **Mean Squared Error (MSE)**: The average of the *squared* differences. Heavily penalizes large errors. Not in the original unit of measurement (e.g., "Dollars squared").
- **Root Mean Squared Error (RMSE)**: The square root of MSE. Brings the error back to the original unit (e.g., "$"). Best default metric because it penalizes large errors while remaining interpretable.
- **R-squared ($R^2$)**: The proportion of variance in the target that can be explained by the features. 1.0 is perfect, 0.0 is predicting the mean, negative means worse than predicting the mean.
- **Adjusted R-squared**: Adjusts $R^2$ based on the number of features. Prevents you from artificially inflating $R^2$ by just throwing useless features at the model.

## Core Tuning Concepts
- **Hyperparameters**: Settings you configure *before* training (like `max_depth` in trees or `alpha` in Ridge). The model does *not* learn these.
- **Grid Search**: Trying every single combination of hyperparameters in a defined grid. Exhaustive but very slow.
- **Randomized Search**: Trying a random sample of combinations. Much faster and often finds a "good enough" solution.
- **Cross-Validation (CV)**: Splitting the training data into $K$ folds. Train on $K-1$, validate on the remaining 1. Repeat $K$ times. Ensures your tuning results aren't just lucky on one specific train/test split.

## Real-World Uses
- Choosing between a fast Linear model and a slow Random Forest by comparing their RMSE and prediction latency.
- Spending compute time tuning an XGBoost model to squeeze out an extra 1% accuracy in a high-frequency trading algorithm where 1% equals millions of dollars.

## Common Mistakes
- Relying entirely on $R^2$. Always look at MAE or RMSE to understand the physical magnitude of the error.
- Tuning hyperparameters on the Test Set. This causes "data leakage." You must tune using Cross-Validation on the Training Set, and only evaluate on the Test Set at the very end.
- Searching too broad a grid with Grid Search, causing the code to run for days.

## Code References
- `code/example-01-metrics.py`
- `code/example-02-gridsearch.py`


---

## Method Options: Evaluation and Tuning

### Regression Metrics in `sklearn.metrics`

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

### `sklearn.model_selection.GridSearchCV`

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

### `sklearn.model_selection.RandomizedSearchCV`

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

---

## Practice: Evaluation and Tuning

### Conceptual Questions
1. You are predicting patient wait times in an ER. An R-squared of 0.82 is reported. Your hospital manager asks "But how many minutes off are we usually?" Which metric should you look at to answer this?
2. Why is it important to use Cross-Validation (CV) during Hyperparameter Tuning instead of just a single Train/Validation split?
3. In `GridSearchCV`, why is the scoring parameter for MSE passed as `neg_mean_squared_error`?

### Coding Tasks

#### Task 1: Understanding Metrics
1. Load a dataset and split it into train/test.
2. Train a `LinearRegression` model.
3. Calculate and print the MAE, MSE, RMSE, and R2 score.
4. Add an extreme outlier to your test predictions (e.g., change one prediction to be $1,000,000 off).
5. Recalculate the metrics. Which metric (MAE or RMSE) changed the most violently due to the single outlier? Why?

#### Task 2: Implementing RandomizedSearchCV
1. Load a dataset and train a standard `RandomForestRegressor`. Record its Test RMSE.
2. Set up a `RandomizedSearchCV` to tune:
   - `n_estimators`: between 50 and 300
   - `max_depth`: [None, 5, 10, 15]
   - `min_samples_split`: [2, 5, 10]
3. Run the search with `n_iter=10` and `cv=3`.
4. Print `search.best_params_`.
5. Extract `search.best_estimator_` and evaluate its Test RMSE. Did tuning improve the model?

---

## Interview Questions: Evaluation and Tuning

### Beginner Level
1. **What is the difference between MAE and RMSE?**
   - *Expected Answer*: MAE is the simple average of the absolute errors. RMSE squares the errors before averaging them, and then takes the square root. Because it squares the errors, RMSE heavily penalizes large errors compared to MAE.

2. **What does an R-squared of 0 mean? What about a negative R-squared?**
   - *Expected Answer*: An R-squared of 0 means the model is no better than just predicting the mean (average) of the target variable for every single observation. A negative R-squared means the model is actually performing *worse* than a flat line predicting the mean.

### Intermediate Level
3. **What is Adjusted R-squared and why is it preferred over R-squared in Multiple Linear Regression?**
   - *Expected Answer*: Normal R-squared will either stay the same or increase every time you add a new feature, even if the feature is total garbage. Adjusted R-squared penalizes the model for adding features that don't genuinely improve predictions, preventing you from artificially inflating the score.

4. **Explain how K-Fold Cross-Validation works.**
   - *Expected Answer*: The training data is split into K equal parts (folds). The model is trained on K-1 folds and validated on the 1 remaining fold. This process is repeated K times, with each fold serving as the validation set exactly once. The final validation score is the average of the K scores.

### Advanced / Practical Level
5. **If you have a massive dataset (10 million rows) and 15 hyperparameters to tune, would you use GridSearchCV? Why or why not?**
   - *Expected Answer*: Absolutely not. Grid search evaluates every single possible combination. With that much data and that many parameters, it would take weeks or months to compute. I would use RandomizedSearchCV to test a random subset of combinations, or Bayesian Optimization (like Optuna) to intelligently search the space.

6. **What is "Data Leakage" during tuning?**
   - *Expected Answer*: It occurs when you tune hyperparameters using the Test Set. If you try 100 combinations and pick the one that scores highest on the Test Set, your model hasn't truly generalized—you've just manually selected the parameters that memorize that specific Test Set. The Test Set must remain completely unseen until the final evaluation.

---

## Python Code Examples

### `example-01-metrics.py`

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Simulate Actual vs Predicted values
y_actual = np.array([100, 150, 200, 250, 300])

# Model A is consistently off by a little bit
y_pred_A = np.array([110, 160, 210, 260, 310])

# Model B is mostly perfect, but makes one HUGE mistake
y_pred_B = np.array([100, 150, 200, 250, 350]) 

# 2. Function to print all metrics
def print_metrics(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    
    print(f"--- {model_name} ---")
    print(f"MAE:  {mae:.2f} (Average absolute error)")
    print(f"MSE:  {mse:.2f} (Average squared error - units are squared!)")
    print(f"RMSE: {rmse:.2f} (Root of MSE - penalizes large errors)")
    print(f"R2:   {r2:.4f} (Variance explained)")
    print("")

# 3. Compare the models
print_metrics("Model A (Consistent small errors)", y_actual, y_pred_A)
print_metrics("Model B (One huge error)", y_actual, y_pred_B)

print("Key Takeaway:")
print("Model A and Model B both have a MAE of 10.")
print("HOWEVER, Model B has a much worse RMSE (22.36 vs 10.00).")
print("This shows how RMSE heavily penalizes large errors (the single error of 50 in Model B).")
```

### `example-02-gridsearch.py`

```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

# 1. Load Data
california = fetch_california_housing()
# Using a small subset of data just to make the grid search run quickly for this example
X = pd.DataFrame(california.data, columns=california.feature_names).iloc[:1000]
y = california.target[:1000]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define the Base Model
rf = RandomForestRegressor(random_state=42)

# 3. Define the Grid of Hyperparameters to search
# It will test 3 x 3 x 2 = 18 different models
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 10]
}

# 4. Set up GridSearchCV
print("Starting Grid Search... this will train 18 models x 3 CV folds = 54 models.")
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3, # 3-fold cross-validation
    scoring='neg_mean_squared_error', # We want to minimize error
    n_jobs=-1, # Use all processor cores
    verbose=1 # Print progress
)

# 5. Run the Search (Warning: This takes time on real datasets!)
grid_search.fit(X_train, y_train)

# 6. View the Results
print("\n--- Tuning Results ---")
print(f"Best Parameters Found: {grid_search.best_params_}")

# Notice how we take the absolute value and square root of the negative MSE
best_cv_rmse = (-grid_search.best_score_) ** 0.5
print(f"Best CV RMSE: {best_cv_rmse:.4f}")

# 7. Evaluate on the totally unseen Test Set
# grid_search automatically refits the best model on the entire training set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"\nFinal Test RMSE (unseen data): {test_rmse:.4f}")
```

### `example-03-randomizedsearch-cv.py`

```python
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import fetch_california_housing
from scipy.stats import randint

# 1. Load Data
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names).iloc[:5000]
y = california.target[:5000]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Setup Random Forest and parameter distributions
rf = RandomForestRegressor(random_state=42)

# Using scipy.stats.randint to define a continuous distribution to sample from
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 15),
    'max_features': ['sqrt', 'log2', 1.0]
}

# 3. Setup RandomizedSearchCV
print("Running Randomized Search (trying 15 combinations)...")
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=15, # Sample 15 random combinations
    cv=3, # 3-fold cross validation
    scoring='neg_mean_absolute_error', # We want to minimize MAE
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)

# 4. Results
print("\n--- Tuning Results ---")
print(f"Best Parameters: {random_search.best_params_}")

# Notice how we take the absolute value of the negative MAE
best_cv_mae = abs(random_search.best_score_)
print(f"Best CV MAE: ${best_cv_mae * 100000:.2f}")

# 5. Evaluate on Test Set
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)

print(f"\nFinal Test MAE (unseen data): ${test_mae * 100000:.2f}")
print("Randomized search is much faster than grid search when the parameter space is large!")
```
