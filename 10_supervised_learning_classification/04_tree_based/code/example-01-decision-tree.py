"""
Example 01: The Overfitting Problem of a Single Decision Tree
Goal: Prove that an unconstrained Decision Tree will overfit, and show how to fix it.
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Create a slightly complex synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- EXPERIMENT 1: Unconstrained Tree ---
# We do not set max_depth. The tree will grow until every leaf is pure.
tree_overfit = DecisionTreeClassifier(random_state=42)
tree_overfit.fit(X_train, y_train)

print("--- Unconstrained Decision Tree ---")
print(f"Training Accuracy: {accuracy_score(y_train, tree_overfit.predict(X_train)) * 100}%")
print(f"Testing Accuracy:  {accuracy_score(y_test, tree_overfit.predict(X_test)) * 100:.2f}%")
print(f"Depth of the tree: {tree_overfit.get_depth()}")
print("Notice the 100% training accuracy. The tree memorized the training data, leading to worse test accuracy.\n")

# --- EXPERIMENT 2: Constrained Tree (Pruning) ---
# We limit the depth to 4.
tree_pruned = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_pruned.fit(X_train, y_train)

print("--- Constrained Decision Tree (max_depth=4) ---")
print(f"Training Accuracy: {accuracy_score(y_train, tree_pruned.predict(X_train)) * 100:.2f}%")
print(f"Testing Accuracy:  {accuracy_score(y_test, tree_pruned.predict(X_test)) * 100:.2f}%")
print("By stopping the tree from growing too deep, it generalized better to the unseen test data.")
