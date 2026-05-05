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
