import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 1. Generate Non-Linear Data (e.g., Company Growth over 10 years)
np.random.seed(42)
years = np.arange(1, 11).reshape(-1, 1) # X
# Revenue grows quadratically, with some noise
revenue = 10 + 2 * (years**2) + np.random.normal(0, 15, size=(10, 1)) # y

# 2. Try fitting a standard straight line first (to see it fail)
linear_model = LinearRegression()
linear_model.fit(years, revenue)
y_pred_linear = linear_model.predict(years)

# 3. Apply Polynomial Transformation
# Create a transformer that will add x^2 features
poly = PolynomialFeatures(degree=2, include_bias=False)
years_poly = poly.fit_transform(years)

# Print to show students what happened
print("Original X (Years):\n", years[:3])
print("\nTransformed X (Years, Years^2):\n", years_poly[:3])

# 4. Fit a Linear Regression model on the Transformed Data
poly_model = LinearRegression()
poly_model.fit(years_poly, revenue)
y_pred_poly = poly_model.predict(years_poly)

# 5. Visualize the difference
plt.figure(figsize=(10, 6))
plt.scatter(years, revenue, color='black', label='Actual Data')
plt.plot(years, y_pred_linear, color='red', linestyle='--', label='Standard Linear Fit (Underfit)')
plt.plot(years, y_pred_poly, color='blue', linewidth=2, label='Polynomial Fit (Degree 2)')

plt.title('Company Revenue Growth: Linear vs Polynomial Fit')
plt.xlabel('Years')
plt.ylabel('Revenue (Millions)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Demonstrate what happens if we extrapolate (predict year 15)
year_15 = np.array([[15]])
year_15_poly = poly.transform(year_15)

pred_linear = linear_model.predict(year_15)[0][0]
pred_poly = poly_model.predict(year_15_poly)[0][0]

print(f"\nPrediction for Year 15:")
print(f"Linear Model Predicts: ${pred_linear:.2f} Million")
print(f"Polynomial Model Predicts: ${pred_poly:.2f} Million")
