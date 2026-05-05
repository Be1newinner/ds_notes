# Practice: Evaluation and Tuning

## Conceptual Questions
1. You are predicting patient wait times in an ER. An R-squared of 0.82 is reported. Your hospital manager asks "But how many minutes off are we usually?" Which metric should you look at to answer this?
2. Why is it important to use Cross-Validation (CV) during Hyperparameter Tuning instead of just a single Train/Validation split?
3. In `GridSearchCV`, why is the scoring parameter for MSE passed as `neg_mean_squared_error`?

## Coding Tasks

### Task 1: Understanding Metrics
1. Load a dataset and split it into train/test.
2. Train a `LinearRegression` model.
3. Calculate and print the MAE, MSE, RMSE, and R2 score.
4. Add an extreme outlier to your test predictions (e.g., change one prediction to be $1,000,000 off).
5. Recalculate the metrics. Which metric (MAE or RMSE) changed the most violently due to the single outlier? Why?

### Task 2: Implementing RandomizedSearchCV
1. Load a dataset and train a standard `RandomForestRegressor`. Record its Test RMSE.
2. Set up a `RandomizedSearchCV` to tune:
   - `n_estimators`: between 50 and 300
   - `max_depth`: [None, 5, 10, 15]
   - `min_samples_split`: [2, 5, 10]
3. Run the search with `n_iter=10` and `cv=3`.
4. Print `search.best_params_`.
5. Extract `search.best_estimator_` and evaluate its Test RMSE. Did tuning improve the model?
