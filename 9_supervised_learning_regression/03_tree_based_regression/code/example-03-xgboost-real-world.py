import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
import xgboost as xgb
import matplotlib.pyplot as plt

# 1. Load Data
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = california.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train XGBoost Regressor
# XGBoost has many hyperparameters. Here are the most common ones.
print("Training XGBoost Model...")
xgb_model = xgb.XGBRegressor(
    n_estimators=300,       # Number of trees
    learning_rate=0.05,     # Step size shrinkage
    max_depth=6,            # Maximum depth of each tree
    subsample=0.8,          # Use 80% of data per tree (prevents overfitting)
    colsample_bytree=0.8,   # Use 80% of features per tree
    random_state=42,
    n_jobs=-1
)

# We can use early stopping to halt training if the validation score stops improving
# Note: eval_set is used as the validation set during training
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50 # Print progress every 50 trees
)

# 3. Evaluate the Final Model
y_pred = xgb_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("\n--- XGBoost Model Evaluation ---")
print(f"R-squared: {r2:.4f}")
print(f"RMSE: ${rmse * 100000:.2f}")

# 4. Plot Training History (Learning Curve)
# Extract the evaluation results
results = xgb_model.evals_result()
val_rmse = results['validation_0']['rmse']

plt.figure(figsize=(10, 5))
plt.plot(val_rmse, label='Validation RMSE')
plt.title('XGBoost Learning Curve')
plt.xlabel('Number of Trees (Boosting Rounds)')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\nNotice how the error drops quickly at first, then starts to plateau.")
print("If we trained for 10,000 trees, the model would likely overfit, and validation RMSE would start rising again.")
