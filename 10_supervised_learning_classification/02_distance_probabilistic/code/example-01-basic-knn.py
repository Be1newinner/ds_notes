"""
Example 01: Basic K-Nearest Neighbors (KNN)
Goal: Understand the mechanics of KNN and prove why scaling features is mandatory.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Create a synthetic dataset with drastically different scales
# Imagine predicting Loan Approval based on Age and Income
data = {
    'Age': [25, 30, 35, 40, 45, 50, 55, 60],
    'Income': [40000, 50000, 60000, 150000, 160000, 170000, 180000, 190000],
    'Approved': [0, 0, 0, 1, 1, 1, 1, 1]  # 0 = No, 1 = Yes
}
df = pd.DataFrame(data)

X = df[['Age', 'Income']]
y = df['Approved']

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- EXPERIMENT 1: NO SCALING ---
knn_unscaled = KNeighborsClassifier(n_neighbors=3)
knn_unscaled.fit(X_train, y_train)
pred_unscaled = knn_unscaled.predict(X_test)
acc_unscaled = accuracy_score(y_test, pred_unscaled)

print(f"Accuracy without scaling: {acc_unscaled * 100:.2f}%")
print("Why? The Income feature ranges from 40k to 190k, while Age ranges from 25 to 60.")
print("The Euclidean distance calculation completely ignores Age because the Income difference is massive.\n")

# --- EXPERIMENT 2: WITH SCALING ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_scaled = KNeighborsClassifier(n_neighbors=3)
knn_scaled.fit(X_train_scaled, y_train)
pred_scaled = knn_scaled.predict(X_test_scaled)
acc_scaled = accuracy_score(y_test, pred_scaled)

print(f"Accuracy WITH scaling: {acc_scaled * 100:.2f}%")
print("Now both Age and Income contribute equally to finding the 'nearest' neighbors.")
