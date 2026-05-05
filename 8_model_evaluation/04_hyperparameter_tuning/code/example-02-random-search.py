"""
Example 02: Random Search
This script demonstrates how Random Search can find a great model much faster than Grid Search.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
import time

# 1. Load Data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define Distributions instead of discrete lists
# If we used Grid Search on ranges this big, it would literally take days to run.
param_dist = {
    'n_estimators': randint(50, 500),    # Any random integer between 50 and 500
    'max_depth': randint(3, 50),         # Any random integer between 3 and 50
    'min_samples_split': randint(2, 20)  # Any random integer between 2 and 20
}

# 3. Initialize Random Search
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,          # ONLY try 20 random combinations!
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

# 4. Run and Time it
print("Running Random Search (20 iterations)...")
start_time = time.time()
random_search.fit(X_train, y_train)
end_time = time.time()

# 5. Results
print(f"Finished in {end_time - start_time:.2f} seconds")
print(f"Best Hyperparameters: {random_search.best_params_}")

best_model = random_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Tuned Model Test Score: {test_score * 100:.2f}%")

# Why this matters:
# Random Search explored a massive space (hundreds of thousands of possible combinations)
# but only actually trained 20 models. It almost always finds a "very good" solution 
# in a tiny fraction of the time it takes Grid Search to find the "perfect" solution.
