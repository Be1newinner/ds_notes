"""
Example 03: Real-World Scenario - Medical Diagnosis
Goal: Apply SVM to a high-dimensional tabular dataset, utilizing scaling and probability outputs.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. Load Breast Cancer Dataset (30 numerical features)
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target  # 0 = Malignant, 1 = Benign

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. CRITICAL: Scale the data
# SVMs calculate distance. Without scaling, features with large ranges dominate.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Initialize Model with probability=True
# probability=True makes training slower, but allows us to use .predict_proba() later
model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)

# 4. Train
model.fit(X_train_scaled, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_scaled)
print("--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# 6. View Probabilities
probs = model.predict_proba(X_test_scaled)
print("\n--- Probability Output for First 3 Patients ---")
for i in range(3):
    pred_class = cancer.target_names[y_pred[i]]
    prob_benign = probs[i][1] * 100
    prob_malignant = probs[i][0] * 100
    print(f"Patient {i+1}: Predicted {pred_class}")
    print(f"  Confidence -> Benign: {prob_benign:.1f}% | Malignant: {prob_malignant:.1f}%")
