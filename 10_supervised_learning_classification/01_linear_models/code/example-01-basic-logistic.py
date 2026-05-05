"""
Example 01: Basic Logistic Regression
Goal: Understand how to fit a basic Logistic Regression model for binary classification.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Create a simple synthetic dataset
# Scenario: Predicting if a student passes a test based on Hours Studied
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Passed': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}
df = pd.DataFrame(data)

# 2. Separate features (X) and target (y)
X = df[['Hours_Studied']]
y = df['Passed']

# 3. Initialize the model
# We use default settings (L2 regularization, C=1.0)
model = LogisticRegression()

# 4. Train the model
model.fit(X, y)

# 5. Make predictions on some new student data
new_students = pd.DataFrame({'Hours_Studied': [2.5, 4.5, 6.5]})
predictions = model.predict(new_students)

# predict_proba gives the actual probabilities [prob_Fail, prob_Pass]
probabilities = model.predict_proba(new_students)

print("--- Prediction Results ---")
for i, hours in enumerate(new_students['Hours_Studied']):
    prob_pass = probabilities[i][1] * 100
    pred_class = "Pass" if predictions[i] == 1 else "Fail"
    print(f"Hours Studied: {hours} | Prob of Passing: {prob_pass:.2f}% | Final Prediction: {pred_class}")

# 6. Look at the learned coefficients
print("\n--- Model Internals ---")
print(f"Coefficient (Weight for Hours_Studied): {model.coef_[0][0]:.4f}")
print(f"Intercept (Bias): {model.intercept_[0]:.4f}")
print("Notice the coefficient is positive: More hours studied increases the probability of passing.")
