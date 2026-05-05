"""
Example 02: Random Forest Classification
Goal: Use the industry-standard Random Forest to improve upon a single tree's performance.
"""

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the Wine Quality dataset (Predicting 3 different classes of wine)
wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Initialize Random Forest
# n_estimators = 100 means we are building 100 different decision trees.
# n_jobs = -1 tells the computer to build these trees in parallel using all CPU cores.
rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

# 3. Train the model
# NOTE: We did NOT scale the data. Tree models do not require scaling!
rf_model.fit(X_train, y_train)

# 4. Evaluate
y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Random Forest Accuracy: {acc * 100:.2f}%\n")
print("--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

print("Random Forest almost always beats a single decision tree out-of-the-box.")
