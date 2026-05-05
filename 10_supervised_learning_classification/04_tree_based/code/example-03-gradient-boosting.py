"""
Example 03: Gradient Boosting
Goal: Implement a sequential ensemble method and understand the learning rate.
"""

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# 1. Load Data (same as Random Forest example for comparison)
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=42)

# 2. Experiment with different Learning Rates
learning_rates = [1.0, 0.1, 0.01]

print("--- Gradient Boosting with Different Learning Rates ---")
for lr in learning_rates:
    # Initialize Gradient Boosting
    # max_depth=3 (Boosting works best with very shallow trees)
    gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=lr, max_depth=3, random_state=42)
    
    # Train
    gbm.fit(X_train, y_train)
    
    # Predict & Evaluate
    train_acc = accuracy_score(y_train, gbm.predict(X_train))
    test_acc = accuracy_score(y_test, gbm.predict(X_test))
    
    print(f"Learning Rate: {lr:5} | Train Acc: {train_acc*100:6.2f}% | Test Acc: {test_acc*100:6.2f}%")

print("\nTakeaway:")
print("A learning rate of 1.0 (too fast) overfits the data.")
print("A learning rate of 0.1 provides a good balance.")
print("A learning rate of 0.01 (too slow) underfits because 100 trees aren't enough at that speed.")
