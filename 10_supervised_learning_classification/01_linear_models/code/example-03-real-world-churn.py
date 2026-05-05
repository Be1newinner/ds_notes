"""
Example 03: Real-World Scenario - Customer Churn Prediction
Goal: Apply Logistic Regression to a business problem, including data scaling and coefficient interpretation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# 1. Simulate a Telecom Churn Dataset
np.random.seed(42)
n_samples = 1000

# Features: Tenure (months), Monthly Charge ($), Calls to Customer Service
tenure = np.random.randint(1, 72, n_samples)
monthly_charge = np.random.uniform(20, 120, n_samples)
customer_service_calls = np.random.randint(0, 6, n_samples)

# Create a "churn probability" driven by the features
# Lower tenure, higher charge, more service calls = higher chance of churn
log_odds = -0.05 * tenure + 0.02 * monthly_charge + 0.8 * customer_service_calls - 2
prob_churn = 1 / (1 + np.exp(-log_odds))
churn = (np.random.rand(n_samples) < prob_churn).astype(int)

df = pd.DataFrame({
    'Tenure_Months': tenure,
    'Monthly_Charge': monthly_charge,
    'Cust_Service_Calls': customer_service_calls,
    'Churn': churn
})

X = df.drop('Churn', axis=1)
y = df['Churn']

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. CRITICAL STEP: Scale the features
# Logistic regression uses L2 regularization by default, so scaling is mandatory!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train the model
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_scaled)
print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# 6. Business Interpretation of Coefficients
print("\n--- Feature Importance (Coefficients) ---")
# Because data is scaled, the magnitude of the coefficient shows importance
for feature, coef in zip(X.columns, model.coef_[0]):
    direction = "Increases" if coef > 0 else "Decreases"
    print(f"{feature:20} : {coef:>7.4f} ({direction} chance of Churn)")

print("\nInsight for Business Team:")
print("Customers with many customer service calls have the highest risk of churning.")
print("Customers who have been with us longer (higher tenure) have a lower risk of churning.")
