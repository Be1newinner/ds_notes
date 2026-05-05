"""
Example 02: R-squared and Adjusted R-squared
This script shows how standard R-squared can be tricked by useless data.
"""

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# Helper function to calculate Adjusted R-squared
def adjusted_r2(r2, n_samples, n_features):
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

# 1. Base Model with 5 Good Features
X_good, y = make_regression(n_samples=200, n_features=5, noise=15, random_state=42)
X_train_g, X_test_g, y_train, y_test = train_test_split(X_good, y, test_size=0.2, random_state=42)

model_g = LinearRegression().fit(X_train_g, y_train)
y_pred_g = model_g.predict(X_test_g)

r2_g = r2_score(y_test, y_pred_g)
adj_r2_g = adjusted_r2(r2_g, len(y_test), X_test_g.shape[1])

print("--- MODEL WITH 5 USEFUL FEATURES ---")
print(f"R-squared:          {r2_g:.4f}")
print(f"Adjusted R-squared: {adj_r2_g:.4f}")
print()

# 2. Deceptive Model with 5 Good Features + 50 Useless Features (Random Noise)
X_noise = np.random.rand(200, 50)
X_bad = np.hstack([X_good, X_noise]) # Combine them
X_train_b, X_test_b, y_train, y_test = train_test_split(X_bad, y, test_size=0.2, random_state=42)

model_b = LinearRegression().fit(X_train_b, y_train)
y_pred_b = model_b.predict(X_test_b)

r2_b = r2_score(y_test, y_pred_b)
adj_r2_b = adjusted_r2(r2_b, len(y_test), X_test_b.shape[1])

print("--- MODEL WITH 50 ADDED USELESS FEATURES ---")
print(f"R-squared:          {r2_b:.4f}  <-- Notice how it barely changed, or even went up!")
print(f"Adjusted R-squared: {adj_r2_b:.4f}  <-- Notice how it crashed. It knows the new features are garbage.")
