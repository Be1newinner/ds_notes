"""
Example 03: The ROC-AUC Score
This script demonstrates how to evaluate a model independent of its threshold.
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 1. Create Data and Train Model
X, y = make_classification(n_samples=1000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# 2. Getting Probabilities (CRITICAL STEP)
# ROC-AUC needs the raw probability of the positive class (class 1), NOT the hard 0/1 prediction.
# predict_proba returns two columns: [Probability of 0, Probability of 1]
# We want the second column (index 1).
y_pred_probabilities = model.predict_proba(X_test)[:, 1]

# 3. Calculate AUC
auc_score = roc_auc_score(y_test, y_pred_probabilities)

print(f"ROC-AUC Score: {auc_score:.3f}")
print("Interpretation:")
print("0.500 = Random Guessing")
print("0.700 = Acceptable")
print("0.800 = Excellent")
print("1.000 = Perfect Model")
