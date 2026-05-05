"""
Example 02: Randomized Search CV
Goal: Use RandomizedSearchCV to tune a Random Forest quickly.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
import time

# 1. Load Data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=42)

# 2. Define the model
rf = RandomForestClassifier(random_state=42)

# 3. Define Parameter Distributions
# Unlike GridSearch, we can define continuous ranges using scipy.stats
param_dist = {
    'n_estimators': randint(50, 200),      # Any random integer between 50 and 200
    'max_depth': [None, 5, 10, 15, 20],    # Discrete list
    'min_samples_split': randint(2, 11)    # Any random integer between 2 and 10
}

# 4. Initialize RandomizedSearchCV
# n_iter=20 means we will randomly sample 20 combinations from the ranges above.
random_search = RandomizedSearchCV(
    estimator=rf, 
    param_distributions=param_dist, 
    n_iter=20, 
    cv=5, 
    n_jobs=-1, 
    random_state=42,
    verbose=1
)

# 5. Run the search and time it
start_time = time.time()
print("Starting Randomized Search...")
random_search.fit(X_train, y_train)
end_time = time.time()

# 6. Results
print(f"\nSearch finished in {end_time - start_time:.2f} seconds.")
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best Cross-Validation Score: {random_search.best_score_ * 100:.2f}%")

# 7. Evaluate on Test Set
best_model = random_search.best_estimator_
test_acc = best_model.score(X_test, y_test)
print(f"Final Test Set Accuracy: {test_acc * 100:.2f}%")
