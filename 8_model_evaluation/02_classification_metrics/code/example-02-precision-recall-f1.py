"""
Example 02: Precision, Recall, and the F1-Score
This script shows why Accuracy fails on imbalanced datasets, and how Precision/Recall fix it.
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# 1. Create a highly imbalanced dataset (e.g., 95% Normal, 5% Fraud)
X, y = make_classification(n_samples=1000, weights=[0.95, 0.05], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Train a very simple, perhaps under-tuned model
model = LogisticRegression()
model.fit(X_train, y_train)

# 3. Predict
y_pred = model.predict(X_test)

# 4. The Accuracy Illusion
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy:  {acc * 100:.1f}%  <-- Looks amazing, right?")

# 5. The Honest Truth
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision * 100:.1f}%  <-- When it says Fraud, it is right {precision*100:.0f}% of the time.")
print(f"Recall:    {recall * 100:.1f}%  <-- But it only caught {recall*100:.0f}% of the actual Fraud cases!")
print(f"F1-Score:  {f1 * 100:.1f}%  <-- The true combined score of the model.")

# Conclusion:
# A model with 95% accuracy might only have a 25% F1-score if it is just guessing "Normal" most of the time!
