# Practice: Tree-Based Regression

## Conceptual Questions
1. Why does a Decision Tree Regressor not require feature scaling like StandardScaling, whereas Linear Regression with Ridge/Lasso does?
2. Explain the difference in how Random Forest and Gradient Boosting handle errors made by trees.
3. If your Random Forest model has a massive gap between Training R2 (0.98) and Test R2 (0.60), what parameter would you adjust first and in what direction?

## Coding Tasks

### Task 1: Overfitting a Decision Tree
1. Load a regression dataset (e.g., California Housing).
2. Train a `DecisionTreeRegressor` with `max_depth=None` (the default).
3. Evaluate the R2 score on the Training Set and the Test Set. (Notice the training set score is likely 1.0 or very close to it).
4. Retrain the model using `max_depth=5`. Compare the new Train/Test scores.

### Task 2: Random Forest Feature Importance
1. Load a dataset with many features.
2. Train a `RandomForestRegressor(n_estimators=100)`.
3. Extract `model.feature_importances_`.
4. Create a Pandas DataFrame mapping feature names to their importance scores.
5. Sort the DataFrame in descending order and use Matplotlib/Seaborn to create a horizontal bar chart of the top 10 most important features.

### Task 3: Introduction to XGBoost
1. Install `xgboost` if not already installed.
2. Train an `XGBRegressor` on a dataset.
3. Experiment with the `learning_rate` parameter (try 0.01, 0.1, and 0.5). How does it affect the final test score?
