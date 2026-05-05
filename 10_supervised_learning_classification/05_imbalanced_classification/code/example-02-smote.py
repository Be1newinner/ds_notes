"""
Example 02: Fixing Imbalance with SMOTE
Goal: Learn how to generate synthetic data properly.
Note: You must have 'imbalanced-learn' installed (pip install imbalanced-learn)
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# Import SMOTE
from imblearn.over_sampling import SMOTE

# 1. Create imbalanced data (90% Class 0, 10% Class 1)
X, y = make_classification(n_samples=1000, n_features=5, weights=[0.90], random_state=42)

# 2. Split the data FIRST (The Golden Rule)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("--- BEFORE SMOTE ---")
print(f"Train Class 0 count: {sum(y_train == 0)}")
print(f"Train Class 1 count: {sum(y_train == 1)}")

# 3. Apply SMOTE to the TRAINING data ONLY
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\n--- AFTER SMOTE ---")
print(f"Train Class 0 count: {sum(y_train_resampled == 0)}")
print(f"Train Class 1 count: {sum(y_train_resampled == 1)}")
print("SMOTE generated synthetic rows so the classes are perfectly equal 50/50.")

# 4. Train model on the resampled data
model = RandomForestClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# 5. Evaluate on the ORIGINAL, UNTOUCHED test data
y_pred = model.predict(X_test)

print("\n--- Classification Report (Evaluated on Untouched Test Set) ---")
print(classification_report(y_test, y_pred))
print("By feeding the model balanced data during training, it learned to recognize Class 1 better.")
