# Examples: Hyperparameter Tuning

Here is a breakdown of the Python examples provided in the `code/` directory.

## 1. Grid Search (`example-01-grid-search.py`)
- **Goal:** Learn how to set up an exhaustive search for the best model parameters.
- **Dataset:** Breast Cancer dataset.
- **Key Concepts Shown:** 
  - Defining a `param_grid` dictionary.
  - Initializing and fitting `GridSearchCV` on a Support Vector Machine.
  - Accessing `best_params_` and evaluating the `best_estimator_`.
- **Takeaway:** Grid Search is simple to set up but can take a long time to run. It automatically handles cross-validation so you don't overfit to a validation set.

## 2. Randomized Search (`example-02-random-search.py`)
- **Goal:** Show a faster alternative to Grid Search when dealing with many parameters.
- **Dataset:** Breast Cancer dataset.
- **Key Concepts Shown:** 
  - Using `scipy.stats` to define continuous distributions of parameters.
  - Using `RandomizedSearchCV` to search through a Random Forest's hyperparameter space.
  - Controlling the time budget using `n_iter`.
- **Takeaway:** Random search is the preferred method when the search space is large or continuous.

## 3. Advanced Workflow with Pipelines (`example-03-advanced-workflow.py`)
- **Goal:** Combine preprocessing and modeling into a single tunable entity.
- **Dataset:** Synthetic classification dataset.
- **Key Concepts Shown:** 
  - Building a `Pipeline` with `StandardScaler` and `LogisticRegression`.
  - Using the double-underscore syntax (`model__C`) to tune parameters inside the pipeline.
  - Why pipelines prevent Data Leakage during cross-validation.
- **Takeaway:** In production code, you should almost always tune `Pipelines`, not naked models. This ensures that scaling and imputation are executed freshly on every cross-validation fold.
