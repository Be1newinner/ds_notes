import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

# 1. Load a real dataset (California Housing)
print("Loading California Housing Data...")
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = california.target # Target is median house value in 100,000s

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Random Forest
# Notice: WE ARE NOT SCALING THE DATA. Trees don't care about scale!
print("Training Random Forest (this might take a few seconds)...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 4. Evaluate
y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("\n--- Model Evaluation ---")
print(f"R-squared: {r2:.4f}")
print(f"RMSE: ${rmse * 100000:.2f}")

# 5. Feature Importance Extraction
print("\n--- Feature Importances ---")
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=True)

# 6. Visualize Feature Importances
plt.figure(figsize=(10, 6))
importances_sorted.plot(kind='barh', color='teal')
plt.title('Random Forest - Feature Importance (California Housing)')
plt.xlabel('Importance Score (Variance Reduction)')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

print("\nInterpretation: 'MedInc' (Median Income in the area) is by far the most important feature for predicting house prices in this model.")
