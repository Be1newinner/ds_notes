# Hyperparameter Tuning: Search Strategies & Workflows

## Learning Objective
Learn how to systematically find the best mathematical settings (hyperparameters) for a machine learning model to maximize its performance without overfitting, using techniques like Grid Search and Random Search.

## What Is This Topic?
When you initialize a model (e.g., `RandomForestClassifier()`), it uses default settings provided by Scikit-Learn. These settings are called **Hyperparameters** (like `max_depth`, `n_estimators`, `learning_rate`). They are settings you configure *before* training. Hyperparameter Tuning is the automated process of trying many different combinations of these settings to find the one that works best for your specific dataset.

*(Note: Parameters, like the weights in Logistic Regression, are learned by the algorithm during training. Hyperparameters are set by the data scientist).*

## Why This Topic Matters
The default settings almost never yield the best possible model. Tuning hyperparameters is how you squeeze the last 5% to 15% of performance out of an algorithm. It is a mandatory step in any professional ML pipeline.

## The Problem with Manual Tuning
You could manually change `max_depth` to 5, train the model, check the score, then change it to 6, train, check, etc. But if you have 4 different hyperparameters to tune, the number of possible combinations explodes. You need an automated strategy.

## Strategy 1: Grid Search
Grid Search is the brute-force approach. You give it a list of values for each hyperparameter (a "grid"). It will train a model for **every single possible combination** and tell you which one was best.
- **Pros**: Guaranteed to find the absolute best combination within the grid you provided.
- **Cons**: Computationally massive. If you have 5 parameters with 5 values each, that's $5^5 = 3,125$ models to train!

## Strategy 2: Randomized Search
Instead of trying every combination, you give it ranges of values, and it randomly selects combinations a set number of times (e.g., "try 100 random combinations").
- **Pros**: Much faster than Grid Search. Statistically, it is highly likely to find a combination that is very close to the optimal one, in a fraction of the time.
- **Cons**: Not guaranteed to find the *absolute* best combination.

## The Role of Cross-Validation (The "CV" in GridSearchCV)
If you tune your model by testing it on your Test Set, you are cheating. You are optimizing the model to specifically pass the test. 
To avoid this, we use **K-Fold Cross-Validation**:
1. Take the Training Set and split it into $K$ chunks (folds), say 5.
2. Train the model on 4 folds, evaluate on the 1 remaining fold.
3. Repeat this 5 times, so every fold gets to be the evaluation set once.
4. Average the 5 scores.
This allows us to evaluate model configurations thoroughly without ever touching the true Test Set.

## Advanced Workflows
In modern Data Science, tuning doesn't just happen on the model. You tune the entire pipeline. For example, you can use Grid Search to find out if `StandardScaler` works better than `MinMaxScaler`, while simultaneously finding the best `C` value for an SVM.

## Code References
- `code/example-01-grid-search.py`
- `code/example-02-random-search.py`
- `code/example-03-advanced-workflow.py`


---

## Method Options: Hyperparameter Tuning in Scikit-Learn

This document explains the search algorithms used for tuning.

### 1. `sklearn.model_selection.GridSearchCV`

#### Syntax
```python
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_dict, cv=5, scoring='accuracy', n_jobs=-1)
```

#### Common Arguments
- **`estimator`**: The model object you want to tune (e.g., `RandomForestClassifier()`).
- **`param_grid`** (`dict` or `list of dicts`): Dictionary with parameters names (string) as keys and lists of parameter settings to try as values.
  - Example: `{'max_depth': [3, 5, 10], 'C': [0.1, 1, 10]}`
- **`cv`** (`int`, default=`5`): Determines the cross-validation splitting strategy. Specifies the number of folds in a `(Stratified)KFold`.
- **`scoring`** (`str`, default=`None`): Strategy to evaluate the performance of the cross-validated model on the test set. For classification, options include `'accuracy'`, `'precision'`, `'recall'`, `'f1'`, `'roc_auc'`. If `None`, it uses the estimator's default `.score()` method.
- **`n_jobs`** (`int`, default=`None`): Number of jobs to run in parallel. `-1` means using all processors. Highly recommended.

#### Useful Attributes After Fitting
- **`best_params_`**: The dictionary of parameters that gave the best results.
- **`best_estimator_`**: The actual model object, refitted on the *entire* training data using the best parameters. You can use this directly to `predict()`.
- **`cv_results_`**: A massive dictionary containing detailed logs of every combination tried and its score.

---

### 2. `sklearn.model_selection.RandomizedSearchCV`

#### Syntax
```python
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats

param_dist = {'C': stats.uniform(0.1, 10), 'max_depth': [3, 5, 10]}
rand_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, cv=5)
```

#### Common Arguments
- **`param_distributions`** (`dict`): Similar to `param_grid`, but values can be lists (discrete choices) or SciPy distributions (continuous ranges). 
- **`n_iter`** (`int`, default=`10`): Number of parameter settings that are sampled. Trades off runtime vs quality of the solution.

---

### 3. `sklearn.pipeline.Pipeline`

Crucial for advanced workflows. A pipeline chains multiple steps (like scaling and modeling) into one object.

#### Syntax
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())
])
```

#### Tuning Pipelines
When using `GridSearchCV` on a `Pipeline`, you must prefix the parameter names with the step name followed by a double underscore (`__`).
- Example: Instead of tuning `'C'`, you must tune `'svc__C'`.

---

## Examples: Hyperparameter Tuning

Here is a breakdown of the Python examples provided in the `code/` directory.

### 1. Grid Search (`example-01-grid-search.py`)
- **Goal:** Learn how to set up an exhaustive search for the best model parameters.
- **Dataset:** Breast Cancer dataset.
- **Key Concepts Shown:** 
  - Defining a `param_grid` dictionary.
  - Initializing and fitting `GridSearchCV` on a Support Vector Machine.
  - Accessing `best_params_` and evaluating the `best_estimator_`.
- **Takeaway:** Grid Search is simple to set up but can take a long time to run. It automatically handles cross-validation so you don't overfit to a validation set.

### 2. Randomized Search (`example-02-random-search.py`)
- **Goal:** Show a faster alternative to Grid Search when dealing with many parameters.
- **Dataset:** Breast Cancer dataset.
- **Key Concepts Shown:** 
  - Using `scipy.stats` to define continuous distributions of parameters.
  - Using `RandomizedSearchCV` to search through a Random Forest's hyperparameter space.
  - Controlling the time budget using `n_iter`.
- **Takeaway:** Random search is the preferred method when the search space is large or continuous.

### 3. Advanced Workflow with Pipelines (`example-03-advanced-workflow.py`)
- **Goal:** Combine preprocessing and modeling into a single tunable entity.
- **Dataset:** Synthetic classification dataset.
- **Key Concepts Shown:** 
  - Building a `Pipeline` with `StandardScaler` and `LogisticRegression`.
  - Using the double-underscore syntax (`model__C`) to tune parameters inside the pipeline.
  - Why pipelines prevent Data Leakage during cross-validation.
- **Takeaway:** In production code, you should almost always tune `Pipelines`, not naked models. This ensures that scaling and imputation are executed freshly on every cross-validation fold.

---

## Practice Exercises: Hyperparameter Tuning

These exercises are designed to test your conceptual understanding and coding skills.

### Conceptual Questions
1. If you run a `GridSearchCV` with 3 values for `max_depth`, 4 values for `n_estimators`, and `cv=5`, exactly how many times will the model be trained?
2. What is Data Leakage in the context of Hyperparameter Tuning? Why do we use cross-validation instead of just tuning on the test set?
3. Why might you prefer `RandomizedSearchCV` over `GridSearchCV` when tuning a Gradient Boosting model with 6 different hyperparameters?

### Coding Tasks

#### Task 1: Basic Grid Search
1. Load the `wine` dataset (`load_wine`). Split into 80/20 train/test.
2. Initialize a `DecisionTreeClassifier`.
3. Create a parameter grid dictionary: `max_depth` from 2 to 10, and `criterion` as `['gini', 'entropy']`.
4. Run a `GridSearchCV` using `cv=3`.
5. Print the `best_params_` and the test accuracy of the best model.

#### Task 2: Changing the Scoring Metric
By default, GridSearchCV optimizes for Accuracy.
1. Generate an imbalanced dataset: `make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)`.
2. Split into train/test.
3. Set up a `GridSearchCV` for a `RandomForestClassifier`. Tune `n_estimators` (10, 50, 100) and `class_weight` (None, 'balanced').
4. **Crucial Step:** Set the `scoring` parameter in GridSearchCV to `'f1'`.
5. Fit the grid search and print the best parameters. Did it choose 'balanced'?

#### Task 3: Pipeline Tuning
1. Load the `breast_cancer` dataset. Split into train/test.
2. Create a `Pipeline` with two steps: `StandardScaler` and `SVC`.
3. Create a parameter grid. You want to test two kernels: `['linear', 'rbf']`, and three values for C: `[0.1, 1.0, 10.0]`. (Remember the double underscore syntax!).
4. Run `GridSearchCV` on the *pipeline*. Print the best parameters.

---

## Interview Questions: Hyperparameter Tuning

### Beginner Questions
1. **What is the difference between a model parameter and a hyperparameter?**
   *Hint:* A model parameter is learned from the data during training (e.g., the weights in Logistic Regression or the splits in a Decision Tree). A hyperparameter is set by the data scientist *before* training begins (e.g., `max_depth` or `learning_rate`).
2. **What does Grid Search do?**
   *Hint:* It takes a list of possible hyperparameter values, creates every possible combination of them, trains a model for each combination, and returns the one that performs best.
3. **What does the "CV" in GridSearchCV stand for, and what does it mean?**
   *Hint:* Cross-Validation. It means the training data is split into multiple folds to evaluate the model safely without touching the final Test Set.

### Conceptual Questions
4. **Why is it considered bad practice to tune hyperparameters by evaluating the model on the final Test Set?**
   *Hint:* This causes Data Leakage. If you tweak the model until it gets a high score on the test set, the model is no longer generalizing to unseen data; it has essentially "memorized" the test set. The test set must be kept strictly separate until the very end.
5. **If you have a massive dataset and 10 hyperparameters to tune, why would you choose Randomized Search over Grid Search?**
   *Hint:* The "Curse of Dimensionality". Grid search time grows exponentially with the number of parameters. Randomized Search samples a fixed number of combinations, meaning it runs much faster and, statistically, often finds a combination very close to the optimal one.
6. **Explain what a Scikit-Learn `Pipeline` is and why it's useful for tuning.**
   *Hint:* A Pipeline bundles data preprocessing steps (like scaling) and the model into a single object. It is useful for tuning because it prevents data leakage during Cross-Validation.

### Practical Questions
7. **During a GridSearchCV with `cv=5`, does the scaler (e.g., StandardScaler) get fit on the entire training set before the folds are created, or is it fit 5 separate times?**
   *Hint:* If you do it correctly (using a Pipeline), the scaler is fit 5 separate times, using only the training folds in each step. If you scale the data *before* running GridSearchCV, you cause data leakage because information from the validation fold leaks into the scaling process.
8. **By default, what metric does `GridSearchCV` try to maximize for classification problems? How do you change it to focus on false negatives?**
   *Hint:* By default, it maximizes Accuracy. To focus on false negatives, you should set `scoring='recall'`.
9. **You run a Grid Search and the best `max_depth` found is 10. Your parameter grid was `[2, 4, 6, 8, 10]`. What should your next step be?**
   *Hint:* Because the best parameter was at the absolute edge of your grid, the true optimal value might be 12 or 15. You should run another search with a grid like `[10, 12, 14, 16]`.

### Output Interpretation
10. **You run a Grid Search. The model with `max_depth=5` scores 92% on the training folds and 91% on the validation folds. The model with `max_depth=15` scores 99% on the training folds and 91.5% on the validation folds. Which model should you choose and why?**
    *Hint:* You should choose `max_depth=5`. While the deeper tree has a tiny bit more validation accuracy, the massive gap between its training and validation score indicates it is severely overfitting. The simpler model is much more robust.

---

## Python Code Examples

### `example-01-grid-search.py`

```python
"""
Example 01: Grid Search CV
Goal: Learn how to exhaustively search for the best model parameters.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd

# 1. Load and prepare data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=42)

# Scale data (Mandatory for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Define the model
model = SVC(random_state=42)

# 3. Define the parameter grid
# We want to test two kernels, and 4 different values of C.
# Total combinations = 2 * 4 = 8.
param_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10, 100]
}

# 4. Initialize GridSearchCV
# cv=5 means 5-fold cross-validation.
# n_jobs=-1 uses all available CPU cores to speed things up.
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

# 5. Run the search
print("Starting Grid Search...")
grid_search.fit(X_train_scaled, y_train)

# 6. Look at the results
print("\n--- Grid Search Results ---")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score (Accuracy): {grid_search.best_score_ * 100:.2f}%")

# 7. Evaluate the best model on the Test Set
# grid_search automatically refits the best model on the entire training data!
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test_scaled, y_test)
print(f"\nFinal Test Set Accuracy: {test_accuracy * 100:.2f}%")

# 8. (Optional) Look at the full results table
results_df = pd.DataFrame(grid_search.cv_results_)
print("\nTop 3 Configurations Tried:")
print(results_df[['param_C', 'param_kernel', 'mean_test_score', 'rank_test_score']].sort_values('rank_test_score').head(3).to_string(index=False))
```

### `example-02-random-search.py`

```python
"""
Example 02: Randomized Search CV
Goal: Use RandomizedSearchCV to tune a Random Forest quickly.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
import time

# 1. Load Data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=42)

# 2. Define the model
rf = RandomForestClassifier(random_state=42)

# 3. Define Parameter Distributions
# Unlike GridSearch, we can define continuous ranges using scipy.stats
param_dist = {
    'n_estimators': randint(50, 200),      # Any random integer between 50 and 200
    'max_depth': [None, 5, 10, 15, 20],    # Discrete list
    'min_samples_split': randint(2, 11)    # Any random integer between 2 and 10
}

# 4. Initialize RandomizedSearchCV
# n_iter=20 means we will randomly sample 20 combinations from the ranges above.
random_search = RandomizedSearchCV(
    estimator=rf, 
    param_distributions=param_dist, 
    n_iter=20, 
    cv=5, 
    n_jobs=-1, 
    random_state=42,
    verbose=1
)

# 5. Run the search and time it
start_time = time.time()
print("Starting Randomized Search...")
random_search.fit(X_train, y_train)
end_time = time.time()

# 6. Results
print(f"\nSearch finished in {end_time - start_time:.2f} seconds.")
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best Cross-Validation Score: {random_search.best_score_ * 100:.2f}%")

# 7. Evaluate on Test Set
best_model = random_search.best_estimator_
test_acc = best_model.score(X_test, y_test)
print(f"Final Test Set Accuracy: {test_acc * 100:.2f}%")
```

### `example-03-advanced-workflow.py`

```python
"""
Example 03: Advanced Workflow - Tuning a Pipeline
Goal: Chain scaling and modeling into a Pipeline to prevent Data Leakage during cross-validation.
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# 1. Create a synthetic dataset
X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Create a Pipeline
# A pipeline executes these steps sequentially.
# This ensures that during Cross-Validation, the Scaler is fit ONLY on the training folds.
pipeline = Pipeline([
    ('scaler', StandardScaler()), # Name of step: 'scaler'
    ('svm', SVC(random_state=42)) # Name of step: 'svm'
])

# 3. Create the Parameter Grid
# VERY IMPORTANT: To target a parameter inside the pipeline, you must use:
# step_name__parameter_name  (Notice the double underscore!)
param_grid = {
    # We can even ask the GridSearch to test two completely different Scalers!
    'scaler': [StandardScaler(), MinMaxScaler()],
    
    # Target the 'C' parameter of the 'svm' step
    'svm__C': [0.1, 1.0, 10.0],
    
    # Target the 'kernel' parameter of the 'svm' step
    'svm__kernel': ['linear', 'rbf']
}

# 4. Initialize GridSearchCV (passing the pipeline, not just a model)
grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

# 5. Fit the GridSearch (Notice we pass the raw, unscaled X_train)
# The pipeline handles the scaling internally.
print("Starting Pipeline Tuning...")
grid.fit(X_train, y_train)

# 6. Results
print("\n--- Pipeline Tuning Results ---")
print("Best Parameters Found:")
for key, value in grid.best_params_.items():
    print(f"  {key}: {value}")

print(f"\nBest Cross-Validation Accuracy: {grid.best_score_ * 100:.2f}%")

# 7. Evaluate on raw test data (The pipeline scales it automatically)
test_accuracy = grid.best_estimator_.score(X_test, y_test)
print(f"Final Test Set Accuracy: {test_accuracy * 100:.2f}%")
```
