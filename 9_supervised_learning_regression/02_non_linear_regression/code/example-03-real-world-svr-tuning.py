import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
import numpy as np

# 1. Load Data
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names).iloc[:1500]
y = california.target[:1500]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Crucial: Scale Data for SVR
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

# 3. Setup SVR and RandomizedSearchCV
svr = SVR(kernel='rbf')

param_dist = {
    'C': np.logspace(-2, 2, 5), # [0.01, 0.1, 1.0, 10.0, 100.0]
    'gamma': ['scale', 'auto', 0.1, 1.0],
    'epsilon': [0.01, 0.1, 0.5]
}

print("Running Randomized Search for SVR Tuning...")
random_search = RandomizedSearchCV(
    estimator=svr,
    param_distributions=param_dist,
    n_iter=10, # Try 10 random combinations
    cv=3,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_scaled, y_train_scaled)

print("\n--- Tuning Results ---")
print(f"Best Parameters: {random_search.best_params_}")

# 4. Evaluate the best model
best_svr = random_search.best_estimator_

# Predict on scaled test data
y_pred_scaled = best_svr.predict(X_test_scaled)

# Inverse transform to get real predictions
y_pred_real = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

rmse = mean_squared_error(y_test, y_pred_real, squared=False)
print(f"Final Test RMSE (unscaled): ${rmse * 100000:.2f}")
