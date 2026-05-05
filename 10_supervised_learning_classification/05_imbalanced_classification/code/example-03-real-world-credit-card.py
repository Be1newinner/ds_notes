"""
Example 03: Real-World Scenario - Credit Card Fraud
Goal: Compare different techniques on a highly imbalanced dataset and interpret business metrics.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE

# 1. Simulate extremely imbalanced Credit Card Data (99% / 1%)
np.random.seed(42)
n_samples = 10000

amount = np.random.exponential(scale=50, size=n_samples)
time_since_last_txn = np.random.exponential(scale=10, size=n_samples)

# Very strict logic for Fraud (Class 1)
log_odds = 0.05 * amount - 0.1 * time_since_last_txn - 6
prob_fraud = 1 / (1 + np.exp(-log_odds))
fraud = (np.random.rand(n_samples) < prob_fraud).astype(int)

X = pd.DataFrame({'Amount': amount, 'Time_Since_Last': time_since_last_txn})
y = fraud

print(f"Total Legit (0): {sum(y==0)} | Total Fraud (1): {sum(y==1)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Approach 1: Ignorance (Do Nothing) ---
rf_baseline = RandomForestClassifier(random_state=42)
rf_baseline.fit(X_train, y_train)
pred_baseline = rf_baseline.predict(X_test)
print("\n--- Baseline (No Adjustments) ---")
print(classification_report(y_test, pred_baseline, zero_division=0))

# --- Approach 2: Class Weights ---
rf_weighted = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_weighted.fit(X_train, y_train)
pred_weighted = rf_weighted.predict(X_test)
print("\n--- With Class Weights ---")
print(classification_report(y_test, pred_weighted))

# --- Approach 3: SMOTE ---
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_sm, y_train_sm)
pred_smote = rf_smote.predict(X_test)
print("\n--- With SMOTE ---")
print(classification_report(y_test, pred_smote))

print("\nBusiness Conclusion:")
print("The baseline model missed almost all the fraud. Class Weights improved Recall drastically.")
print("Depending on the algorithm, SMOTE can sometimes outperform Class Weights by providing")
print("physical data points for the trees to split on.")
