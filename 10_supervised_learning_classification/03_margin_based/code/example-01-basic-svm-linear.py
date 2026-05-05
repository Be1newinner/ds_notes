"""
Example 01: Basic Linear Support Vector Machine
Goal: Understand margins, support vectors, and the C parameter on linearly separable data.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Create a synthetic, perfectly separable dataset
# Class 0: x and y are small
# Class 1: x and y are large
X = np.array([
    [1, 1], [2, 2], [1.5, 1.8], [2, 1], # Class 0
    [8, 8], [9, 9], [8.5, 8.8], [9, 8]  # Class 1
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# 2. Train a Linear SVM with a "Hard" Margin (High C)
# C=1000 tells the model: "Do not allow any misclassifications"
model_hard = SVC(kernel='linear', C=1000)
model_hard.fit(X, y)

print("--- Hard Margin SVM ---")
print(f"Accuracy: {accuracy_score(y, model_hard.predict(X)) * 100}%")
print(f"Number of Support Vectors: {model_hard.n_support_} (Total: {sum(model_hard.n_support_)})")
print("The support vectors are the specific points closest to the boundary.")
print(f"Support Vector Coordinates:\n{model_hard.support_vectors_}")

# 3. Add an "Outlier" that messes up the clean separation
X_noisy = np.vstack((X, [7, 2])) # A Class 1 point mixed in with Class 0
y_noisy = np.append(y, 1)

# 4. Train a Linear SVM with a "Soft" Margin (Low C)
# C=0.01 tells the model: "It's okay to make a mistake if it means a better overall boundary"
model_soft = SVC(kernel='linear', C=0.01)
model_soft.fit(X_noisy, y_noisy)

print("\n--- Soft Margin SVM (with noisy data) ---")
print(f"Accuracy: {accuracy_score(y_noisy, model_soft.predict(X_noisy)) * 100}%")
print(f"Number of Support Vectors: {model_soft.n_support_} (Total: {sum(model_soft.n_support_)})")
print("Notice how there are more support vectors now. The margin is wider, so more points fall inside it or cross it.")
