"""
Example 03: Visualizing Residuals
Metrics are just numbers. You must visualize your errors to truly understand them.
"""

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Train a model
housing = fetch_california_housing()
X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# 2. Calculate Residuals (Errors)
residuals = y_test - y_pred

# 3. Create Visualizations
plt.figure(figsize=(12, 5))

# Plot 1: Actual vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.3)
# Draw the "Perfect Prediction" diagonal line
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted")

# Plot 2: Residual Histogram
plt.subplot(1, 2, 2)
plt.hist(residuals, bins=50, edgecolor='black')
plt.axvline(x=0, color='r', linestyle='--')
plt.xlabel("Prediction Error (Residuals)")
plt.ylabel("Count")
plt.title("Histogram of Residuals")

plt.tight_layout()
plt.savefig("residuals_plot.png")
print("Saved visualization to residuals_plot.png")

# Interpretation:
# The histogram should look like a normal bell curve centered around the red line (Zero error).
# If it is wildly skewed, your model is systematically over-predicting or under-predicting.
