# Practice Exercises: Hyperparameter Tuning

These exercises are designed to test your conceptual understanding and coding skills.

## Conceptual Questions
1. If you run a `GridSearchCV` with 3 values for `max_depth`, 4 values for `n_estimators`, and `cv=5`, exactly how many times will the model be trained?
2. What is Data Leakage in the context of Hyperparameter Tuning? Why do we use cross-validation instead of just tuning on the test set?
3. Why might you prefer `RandomizedSearchCV` over `GridSearchCV` when tuning a Gradient Boosting model with 6 different hyperparameters?

## Coding Tasks

### Task 1: Basic Grid Search
1. Load the `wine` dataset (`load_wine`). Split into 80/20 train/test.
2. Initialize a `DecisionTreeClassifier`.
3. Create a parameter grid dictionary: `max_depth` from 2 to 10, and `criterion` as `['gini', 'entropy']`.
4. Run a `GridSearchCV` using `cv=3`.
5. Print the `best_params_` and the test accuracy of the best model.

### Task 2: Changing the Scoring Metric
By default, GridSearchCV optimizes for Accuracy.
1. Generate an imbalanced dataset: `make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)`.
2. Split into train/test.
3. Set up a `GridSearchCV` for a `RandomForestClassifier`. Tune `n_estimators` (10, 50, 100) and `class_weight` (None, 'balanced').
4. **Crucial Step:** Set the `scoring` parameter in GridSearchCV to `'f1'`.
5. Fit the grid search and print the best parameters. Did it choose 'balanced'?

### Task 3: Pipeline Tuning
1. Load the `breast_cancer` dataset. Split into train/test.
2. Create a `Pipeline` with two steps: `StandardScaler` and `SVC`.
3. Create a parameter grid. You want to test two kernels: `['linear', 'rbf']`, and three values for C: `[0.1, 1.0, 10.0]`. (Remember the double underscore syntax!).
4. Run `GridSearchCV` on the *pipeline*. Print the best parameters.
