"""
Example 03: Nested Cross Validation
This is an advanced technique used when you want the most rigorous, unbiased estimate 
of how your tuned model will perform in the real world.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.svm import SVC

# 1. Load Data
iris = load_iris()
X, y = iris.data, iris.target

# 2. Define the Model and the Grid
model = SVC()
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

# 3. Define the INNER Loop (Used for Tuning)
# This will try the hyperparameters on subsets of the training data
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv)

# 4. Define the OUTER Loop (Used for Evaluation)
# This evaluates the entire tuning process itself!
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 5. Run Nested CV
print("Running Nested Cross-Validation...")
# Notice we are passing the `grid_search` object into `cross_val_score`
# It's a CV inside a CV!
nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv)

print(f"Individual Outer Fold Scores: {nested_scores}")
print(f"Unbiased Expected Performance: {nested_scores.mean() * 100:.2f}% (+/- {nested_scores.std() * 100:.2f}%)")

# Explanation:
# In standard Grid Search, you tune on CV, and evaluate on a static Test Set. 
# But you only have one Test Set! If you got lucky, your reported score might be too high.
# Nested CV splits the data 5 times. Each time, it takes the training portion, 
# runs a FULL Grid Search on it, picks the best model, and evaluates it on the holdout portion.
# The average of those 5 evaluation scores is the most honest estimate of your model's quality.
