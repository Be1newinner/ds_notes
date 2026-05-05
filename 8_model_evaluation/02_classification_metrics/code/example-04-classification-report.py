"""
Example 04: The Classification Report
This is the most common tool used by Data Scientists on a daily basis.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# 1. Load a Multi-Class dataset (Iris has 3 classes of flowers)
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Train a Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 3. Predict
y_pred = model.predict(X_test)

# 4. Print the Classification Report
# We pass target_names so the report uses the real flower names instead of just 0, 1, 2
report = classification_report(y_test, y_pred, target_names=iris.target_names)

print("Classification Report:")
print(report)

# How to read this:
# Support: How many times that flower actually appeared in the test set.
# Precision/Recall for Setosa: It is perfect. It found all of them, and made no mistakes.
# Versicolor vs Virginica: The model sometimes confuses these two. Look at the slight drops in F1-score.
