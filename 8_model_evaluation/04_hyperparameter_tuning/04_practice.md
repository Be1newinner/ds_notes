# Practice Exercises: Hyperparameter Tuning

## Exercise 1: Exploring Parameters
1. Load the `wine` dataset from `sklearn.datasets`.
2. Do a train-test split.
3. Train a standard `RandomForestClassifier()` with no arguments.
4. Calculate the accuracy on the test set.
5. Train a second `RandomForestClassifier(max_depth=2, n_estimators=10)`.
6. Calculate its accuracy. Did it go up or down? By tweaking these numbers, you are manually hyperparameter tuning!

## Exercise 2: Implementing Grid Search
1. Using the data from Exercise 1, define a parameter grid:
   `param_grid = {'max_depth': [2, 5, None], 'n_estimators': [10, 50, 100], 'criterion': ['gini', 'entropy']}`
2. How many total combinations does this grid have?
3. Initialize `GridSearchCV(model, param_grid, cv=5)`.
4. Fit the Grid Search on the *Training* data.
5. Print `grid_search.best_params_`. What was the winning combination?
6. Print `grid_search.score(X_test, y_test)` to see how the winning model performs on the holdout test set.

## Exercise 3: Random Search vs Grid Search
1. Create a massive parameter grid for a Random Forest:
   `param_dist = {'max_depth': range(1, 50), 'n_estimators': range(10, 500)}`
2. If you used Grid Search on this, how many combinations would it try? (Hint: $49 \times 490 = 24,010$). Assuming 5-fold CV, that's over 120,000 model trainings! It would take a long time.
3. Instead, use `RandomizedSearchCV` with `n_iter=20`.
4. Fit it to the training data.
5. Did it find a model with a good test score in a fraction of the time?

## Exercise 4: Information Leakage Check
You decide to fill missing values in your dataset with the mean. 
- **Method A**: You fill the missing values using the mean of the entire dataset. Then you use `GridSearchCV`.
- **Method B**: You build a Scikit-Learn `Pipeline` that fills missing values and trains the model. You pass the *Pipeline* into `GridSearchCV`.
**Question**: Why is Method A considered Data Leakage, and why is Method B the correct way to do it? (Hint: Think about what the Validation folds inside the Grid Search are seeing).
