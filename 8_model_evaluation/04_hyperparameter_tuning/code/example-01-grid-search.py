"""
Example 01: Grid Search
This script demonstrates exhaustive hyperparameter tuning using GridSearchCV.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# 1. Load Data
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train a Baseline Model (Default Settings)
baseline_model = RandomForestClassifier(random_state=42)
baseline_model.fit(X_train, y_train)
baseline_score = baseline_model.score(X_test, y_test)
print(f"Baseline Accuracy (Defaults): {baseline_score * 100:.2f}%")
print("-" * 30)

# 3. Define the Grid of Hyperparameters
# We want to test different combinations of these settings
param_grid = {
    'n_estimators': [50, 100, 200],      # Number of trees
    'max_depth': [None, 5, 10],          # Maximum depth of the tree
    'min_samples_split': [2, 5, 10]      # Minimum samples required to split a node
}
# Total Combinations: 3 * 3 * 3 = 27 combinations.
# With 5-Fold CV, it will train 27 * 5 = 135 models in total!

# 4. Initialize and Run Grid Search
print("Running Grid Search... (This might take a few seconds)")
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,               # 5-Fold Cross Validation
    scoring='accuracy', # Optimize for accuracy
    n_jobs=-1           # Use all available CPU cores to speed it up
)

grid_search.fit(X_train, y_train)

# 5. Analyze the Results
print("Grid Search Complete!")
print(f"Best Hyperparameters found: {grid_search.best_params_}")
print(f"Best CV Score (during training): {grid_search.best_score_ * 100:.2f}%")

# 6. Evaluate the winning model on the unseen Test Set
best_model = grid_search.best_estimator_
tuned_score = best_model.score(X_test, y_test)

print("-" * 30)
print(f"Tuned Model Accuracy (Test Set): {tuned_score * 100:.2f}%")

# Note: Sometimes the baseline is hard to beat! But Grid Search guarantees 
# you have the best possible model from the options you provided.
