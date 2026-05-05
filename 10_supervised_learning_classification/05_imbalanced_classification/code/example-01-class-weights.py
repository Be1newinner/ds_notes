"""
Example 01: Fixing Imbalance with Class Weights
Goal: Show how class_weight='balanced' changes the focus of the model without altering data.
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Create highly imbalanced data (95% Class 0, 5% Class 1)
X, y = make_classification(n_samples=2000, n_features=10, weights=[0.95], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training Class 0 count: {sum(y_train == 0)}")
print(f"Training Class 1 count: {sum(y_train == 1)}")

# --- EXPERIMENT 1: Standard Model ---
model_standard = LogisticRegression(random_state=42)
model_standard.fit(X_train, y_train)
y_pred_std = model_standard.predict(X_test)

print("\n--- Standard Logistic Regression ---")
print(classification_report(y_test, y_pred_std))
print("Notice how the Recall for Class 1 is very low. The model ignores it to keep high overall accuracy.")

# --- EXPERIMENT 2: Balanced Model ---
# We tell the algorithm to penalize Class 1 mistakes heavily
model_balanced = LogisticRegression(class_weight='balanced', random_state=42)
model_balanced.fit(X_train, y_train)
y_pred_bal = model_balanced.predict(X_test)

print("\n--- Balanced Logistic Regression ---")
print(classification_report(y_test, y_pred_bal))
print("Notice how Recall for Class 1 jumped significantly!")
print("Trade-off: We sacrificed some Precision (more false alarms) to catch the minority class.")
