import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Create a dataset with multiple features
# Predicting employee salary based on experience, performance score, and training hours
np.random.seed(42)
n_samples = 200

experience_years = np.random.uniform(1, 15, n_samples)
performance_score = np.random.uniform(1, 10, n_samples)
training_hours = np.random.uniform(10, 100, n_samples)

# Base salary + 5k per year of exp + 2k per performance point + 100 per training hour + noise
salary = 40000 + (5000 * experience_years) + (2000 * performance_score) + (100 * training_hours) + np.random.normal(0, 5000, n_samples)

df = pd.DataFrame({
    'Experience_Years': experience_years,
    'Performance_Score': performance_score,
    'Training_Hours': training_hours,
    'Salary': salary
})

print("Dataset Preview:")
print(df.head(), "\n")

# 2. Split data into Features (X) and Target (y)
X = df[['Experience_Years', 'Performance_Score', 'Training_Hours']]
y = df['Salary']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the Multiple Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Make Predictions
y_pred = model.predict(X_test)

# 5. Evaluate and Interpret
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"R-squared: {r2:.4f}\n")

print("--- Model Interpretation ---")
print(f"Base Salary (Intercept): ${model.intercept_:.2f}")

# Pair features with their learned coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient (Impact on Salary)': model.coef_
})

print("\nCoefficients:")
print(coefficients)
print("\nInterpretation: For every 1 unit increase in 'Experience_Years', salary is predicted to increase by ~$5000, holding other factors constant.")
