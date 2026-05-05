# Practice: Model Tracking with MLflow

These exercises will test your ability to use MLflow to manage experiments.

## Exercise 1: Basic Manual Logging
1. Write a script that trains a simple `LinearRegression` model on dummy data.
2. Set the MLflow experiment name to "Linear_Regression_Tests".
3. Start an MLflow run.
4. Log the parameter `fit_intercept` (True/False).
5. Calculate the Mean Squared Error (MSE) and log it as a metric.
6. Open the MLflow UI and verify your run was recorded.

## Exercise 2: Comparing Model Types
1. Load the Breast Cancer dataset from `sklearn.datasets`.
2. Set the experiment name to "Breast_Cancer_Classification".
3. Write a loop that iterates over three different models: `LogisticRegression`, `RandomForestClassifier`, and `SVC`.
4. Inside the loop, start an MLflow run.
5. Log the name of the model being used as a parameter (e.g., `mlflow.log_param("model_type", "RandomForest")`).
6. Train the model and calculate accuracy.
7. Log the accuracy metric and log the model artifact using `mlflow.sklearn.log_model()`.
8. Open the UI, select all three runs, and click "Compare" to easily see which model performed best.

## Exercise 3: Hyperparameter Tuning with Autologging
1. Load a regression dataset (e.g., California Housing).
2. Enable scikit-learn autologging (`mlflow.sklearn.autolog()`).
3. Set up a `GridSearchCV` or `RandomizedSearchCV` for a `RandomForestRegressor`. Search over parameters like `n_estimators` and `max_depth`.
4. Run the search.
5. Open the MLflow UI. Look at how autologging automatically captured all the different parameter combinations tried by the GridSearch, as well as the best model.
