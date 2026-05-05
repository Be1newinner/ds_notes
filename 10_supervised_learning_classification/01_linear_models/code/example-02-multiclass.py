"""
Example 02: Multiclass Logistic Regression
Goal: Use Logistic Regression to classify data into more than two categories.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the Iris dataset (3 classes of flowers)
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

print(f"Classes to predict: {target_names}")

# 2. Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize the model
# multi_class='multinomial' tells it to use a true multiclass loss function
# max_iter=200 is used because the default (100) might throw a ConvergenceWarning
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)

# 4. Train
model.fit(X_train, y_train)

# 5. Predict on test set
y_pred = model.predict(X_test)

# 6. Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on test set: {acc * 100:.2f}%\n")

print("--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=target_names))

# Let's look at predict_proba for the first test sample
probs = model.predict_proba(X_test[[0]])
print("\n--- Probability Output for the First Test Sample ---")
print(f"Prob Setosa: {probs[0][0]:.4f}")
print(f"Prob Versicolor: {probs[0][1]:.4f}")
print(f"Prob Virginica: {probs[0][2]:.4f}")
print("Notice that the probabilities sum to 1.0")
