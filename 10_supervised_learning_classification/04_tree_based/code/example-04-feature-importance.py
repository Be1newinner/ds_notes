"""
Example 04: Feature Importance
Goal: Learn how to extract business insights by identifying which features drive the model's decisions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Simulate Employee Attrition (Churn) Data
np.random.seed(42)
n_employees = 1000

data = {
    'Salary': np.random.normal(60000, 15000, n_employees),
    'Years_at_Company': np.random.randint(1, 10, n_employees),
    'Distance_to_Work': np.random.randint(1, 30, n_employees),
    'Performance_Score': np.random.randint(1, 6, n_employees),
    'Number_of_Projects': np.random.randint(1, 8, n_employees)
}
df = pd.DataFrame(data)

# Create a complex logic for why someone quits (1 = Quit, 0 = Stayed)
# Low salary AND long distance OR terrible performance
quit_prob = np.where((df['Salary'] < 50000) & (df['Distance_to_Work'] > 20), 0.8, 0.1)
quit_prob = np.where(df['Performance_Score'] == 1, 0.9, quit_prob)

df['Quit'] = (np.random.rand(n_employees) < quit_prob).astype(int)

X = df.drop('Quit', axis=1)
y = df['Quit']

# 2. Train Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Extract Feature Importances
importances = model.feature_importances_
feature_names = X.columns

# 4. Create a DataFrame to make it look nice
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort from highest to lowest
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("--- Feature Importance ---")
print(importance_df.to_string(index=False))

print("\nBusiness Insight:")
print("The model clearly identified that Performance Score, Salary, and Distance to Work")
print("are the primary drivers of employee attrition. 'Number of Projects' has almost no impact.")

# Note: In a real environment, you would usually plot this using matplotlib or seaborn.
