"""
Example 03: Stratified Splitting
This script demonstrates the solution for highly imbalanced datasets.
"""

import numpy as np
from sklearn.model_selection import train_test_split

# 1. Create a highly imbalanced dataset
# Imagine this is transaction data: 950 normal transactions, 50 fraudulent ones
X = np.random.rand(1000, 5) # 1000 rows, 5 features
y_normal = np.zeros(950)    # 0 = Normal
y_fraud = np.ones(50)       # 1 = Fraud
y = np.concatenate([y_normal, y_fraud])

print(f"Original Data Fraud Rate: {np.mean(y) * 100:.1f}%")
print("-" * 30)

# 2. A BAD Split (Random)
# We might accidentally put too few fraud cases in the test set.
X_train_bad, X_test_bad, y_train_bad, y_test_bad = train_test_split(X, y, test_size=0.2, random_state=12)

print("--- WITHOUT Stratification ---")
print(f"Train Set Fraud Rate: {np.mean(y_train_bad) * 100:.1f}%")
print(f"Test Set Fraud Rate:  {np.mean(y_test_bad) * 100:.1f}%")
print("Notice how the Test set doesn't represent the original 5% reality well!")
print()

# 3. A GOOD Split (Stratified)
# We force the split to maintain the exact same ratio of 0s and 1s
X_train_good, X_test_good, y_train_good, y_test_good = train_test_split(X, y, test_size=0.2, random_state=12, stratify=y)

print("--- WITH Stratification ---")
print(f"Train Set Fraud Rate: {np.mean(y_train_good) * 100:.1f}%")
print(f"Test Set Fraud Rate:  {np.mean(y_test_good) * 100:.1f}%")
print("Perfect! Both sets maintain the exact 5% ratio of the original data.")
