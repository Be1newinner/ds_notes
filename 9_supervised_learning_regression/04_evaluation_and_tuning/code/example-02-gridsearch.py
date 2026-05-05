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
