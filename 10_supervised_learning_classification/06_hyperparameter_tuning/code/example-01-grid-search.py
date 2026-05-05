"""
Example 01: Grid Search CV
Goal: Learn how to exhaustively search for the best model parameters.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd

# 1. Load and prepare data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=42)

# Scale data (Mandatory for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Define the model
model = SVC(random_state=42)

# 3. Define the parameter grid
# We want to test two kernels, and 4 different values of C.
# Total combinations = 2 * 4 = 8.
param_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10, 100]
}

# 4. Initialize GridSearchCV
# cv=5 means 5-fold cross-validation.
# n_jobs=-1 uses all available CPU cores to speed things up.
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

# 5. Run the search
print("Starting Grid Search...")
grid_search.fit(X_train_scaled, y_train)

# 6. Look at the results
print("\n--- Grid Search Results ---")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score (Accuracy): {grid_search.best_score_ * 100:.2f}%")

# 7. Evaluate the best model on the Test Set
# grid_search automatically refits the best model on the entire training data!
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test_scaled, y_test)
print(f"\nFinal Test Set Accuracy: {test_accuracy * 100:.2f}%")

# 8. (Optional) Look at the full results table
results_df = pd.DataFrame(grid_search.cv_results_)
print("\nTop 3 Configurations Tried:")
print(results_df[['param_C', 'param_kernel', 'mean_test_score', 'rank_test_score']].sort_values('rank_test_score').head(3).to_string(index=False))
