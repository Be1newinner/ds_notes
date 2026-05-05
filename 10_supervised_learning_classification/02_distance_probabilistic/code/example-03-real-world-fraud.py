"""
Example 03: Real-World Scenario - Tuning K in KNN for Fraud Detection
Goal: Compare KNN and Gaussian Naive Bayes on continuous data, and learn to find the best K.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# 1. Simulate a Credit Card Fraud Dataset (Continuous Data)
np.random.seed(42)
n_samples = 500

# Features: Transaction Amount ($), Distance from Home (miles)
amount = np.random.exponential(scale=100, size=n_samples)
distance = np.random.exponential(scale=50, size=n_samples)

# Make "Fraud" more likely if amount is high AND distance is high
log_odds = 0.02 * amount + 0.05 * distance - 10
prob_fraud = 1 / (1 + np.exp(-log_odds))
fraud = (np.random.rand(n_samples) < prob_fraud).astype(int)

df = pd.DataFrame({'Amount': amount, 'Distance': distance, 'Fraud': fraud})
X = df[['Amount', 'Distance']]
y = df['Fraud']

# 2. Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- EXPERIMENT 1: Gaussian Naive Bayes ---
# We use GaussianNB because our features (Amount, Distance) are continuous numbers
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)
gnb_acc = accuracy_score(y_test, gnb.predict(X_test_scaled))
print(f"Gaussian Naive Bayes Accuracy: {gnb_acc * 100:.2f}%\n")

# --- EXPERIMENT 2: Finding the best K for KNN ---
print("--- Tuning K for K-Nearest Neighbors ---")
k_values = range(1, 30, 2)  # Odd numbers from 1 to 29
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test_scaled))
    accuracies.append(acc)
    print(f"K={k:2} | Accuracy: {acc * 100:.2f}%")

best_k = k_values[np.argmax(accuracies)]
print(f"\nBest K is {best_k} with accuracy {max(accuracies)*100:.2f}%")
print("Notice how accuracy starts low (overfitting), rises, and eventually drops (underfitting).")
