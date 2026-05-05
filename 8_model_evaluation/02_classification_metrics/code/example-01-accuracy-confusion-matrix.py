"""
Example 01: Accuracy and the Confusion Matrix
This script shows how to build and interpret a confusion matrix.
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# 1. Create a synthetic dataset (Binary Classification)
# 1000 samples, roughly 50/50 split of class 0 and class 1
X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)

# 2. Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# 3. Predict
y_pred = model.predict(X_test)

# 4. Evaluate with Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {acc * 100:.1f}%")
print("-" * 30)

# 5. Build the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(f"[{cm[0][0]} (True Negatives)   |  {cm[0][1]} (False Positives)]")
print(f"[{cm[1][0]} (False Negatives)  |  {cm[1][1]} (True Positives)]")

# Interpretation:
# True Negatives (TN): Model correctly said "No"
# False Positives (FP): Model wrongly said "Yes" (False Alarm)
# False Negatives (FN): Model wrongly said "No" (Missed it)
# True Positives (TP): Model correctly said "Yes"
