"""
Example 03: Advanced Workflow - Tuning a Pipeline
Goal: Chain scaling and modeling into a Pipeline to prevent Data Leakage during cross-validation.
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# 1. Create a synthetic dataset
X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Create a Pipeline
# A pipeline executes these steps sequentially.
# This ensures that during Cross-Validation, the Scaler is fit ONLY on the training folds.
pipeline = Pipeline([
    ('scaler', StandardScaler()), # Name of step: 'scaler'
    ('svm', SVC(random_state=42)) # Name of step: 'svm'
])

# 3. Create the Parameter Grid
# VERY IMPORTANT: To target a parameter inside the pipeline, you must use:
# step_name__parameter_name  (Notice the double underscore!)
param_grid = {
    # We can even ask the GridSearch to test two completely different Scalers!
    'scaler': [StandardScaler(), MinMaxScaler()],
    
    # Target the 'C' parameter of the 'svm' step
    'svm__C': [0.1, 1.0, 10.0],
    
    # Target the 'kernel' parameter of the 'svm' step
    'svm__kernel': ['linear', 'rbf']
}

# 4. Initialize GridSearchCV (passing the pipeline, not just a model)
grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

# 5. Fit the GridSearch (Notice we pass the raw, unscaled X_train)
# The pipeline handles the scaling internally.
print("Starting Pipeline Tuning...")
grid.fit(X_train, y_train)

# 6. Results
print("\n--- Pipeline Tuning Results ---")
print("Best Parameters Found:")
for key, value in grid.best_params_.items():
    print(f"  {key}: {value}")

print(f"\nBest Cross-Validation Accuracy: {grid.best_score_ * 100:.2f}%")

# 7. Evaluate on raw test data (The pipeline scales it automatically)
test_accuracy = grid.best_estimator_.score(X_test, y_test)
print(f"Final Test Set Accuracy: {test_accuracy * 100:.2f}%")
