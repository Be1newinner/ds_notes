"""
Example 02: The Kernel Trick
Goal: Show how non-linear kernels (like RBF) can separate data that a linear line cannot.
"""

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Generate "Circles" data
# This creates a dataset where Class 1 is a small circle inside a larger Class 0 circle.
# A straight line CANNOT separate this.
X, y = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("--- Data Generation ---")
print("Data consists of an inner circle and an outer circle.")

# 2. Try to fit a Linear SVM
linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train, y_train)
acc_linear = accuracy_score(y_test, linear_svm.predict(X_test))

print("\n--- Linear Kernel ---")
print(f"Accuracy: {acc_linear * 100:.2f}%")
print("As expected, a straight line fails completely (approx 50% accuracy).")

# 3. Try to fit an RBF (Radial Basis Function) SVM
rbf_svm = SVC(kernel='rbf', C=1.0, gamma='scale')
rbf_svm.fit(X_train, y_train)
acc_rbf = accuracy_score(y_test, rbf_svm.predict(X_test))

print("\n--- RBF Kernel ---")
print(f"Accuracy: {acc_rbf * 100:.2f}%")
print("The RBF kernel projects the data into a higher dimension where a flat plane CAN separate it.")
print("When projected back to 2D, the boundary looks like a circle!")
