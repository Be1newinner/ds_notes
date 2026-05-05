import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Create a simple dataset: Study hours vs. Exam Score
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Exam_Score': [45, 50, 60, 65, 70, 78, 85, 88, 92, 98]
}

df = pd.DataFrame(data)

# Features (X) must be a 2D array, Target (y) is 1D
X = df[['Hours_Studied']] 
y = df['Exam_Score']

# 2. Initialize and Train the Model
model = LinearRegression()
model.fit(X, y)

# 3. Make Predictions
y_pred = model.predict(X)

# 4. Evaluate the Model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Model Intercept (Bias): {model.intercept_:.2f}")
print(f"Model Coefficient (Weight): {model.coef_[0]:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Interpretation: 
# The coefficient tells us how many points the score increases per hour of study.
# The intercept is the predicted score if someone studies for 0 hours.

# 5. Visualize the Line of Best Fit
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line (Best Fit)')
plt.title('Study Hours vs. Exam Score')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Prediction for a new student who studied 7.5 hours
new_hours = pd.DataFrame({'Hours_Studied': [7.5]})
prediction = model.predict(new_hours)
print(f"\nPredicted score for 7.5 hours of study: {prediction[0]:.2f}")
