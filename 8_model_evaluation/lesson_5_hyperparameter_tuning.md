# Hyperparameter Tuning

## Learning Objective
Students will learn the difference between parameters and hyperparameters, and how to programmatically search for the optimal model configuration using Grid Search and Random Search to maximize performance without overfitting.

## What Is This Topic?
Every machine learning model has "settings" (like knobs and dials on a radio) that control how it learns. Hyperparameter tuning is the systematic process of twisting those knobs to find the combination that gives the best possible score on unseen data.

## Why This Topic Matters
The default settings in scikit-learn are usually "okay," but rarely optimal. A model with default settings might achieve 80% accuracy, but the exact same model with properly tuned hyperparameters could achieve 92%. Tuning is the step that takes a model from "functional" to "professional."

## Core Intuition
Imagine baking a cake.
- **Parameters**: The actual ingredients the cake absorbs as it bakes (the patterns the model learns from the data). You don't choose these; the oven (training process) figures it out.
- **Hyperparameters**: The temperature of the oven and the baking time. You *must* set these before you press start. If the temperature is too high, the cake burns (Overfitting). If too low, it's raw (Underfitting).

## Key Concepts

### 1. Parameters vs. Hyperparameters
- **Parameters**: Learned automatically during `model.fit()` (e.g., the slope $m$ and intercept $b$ in Linear Regression).
- **Hyperparameters**: Set manually by the Data Scientist *before* calling `model.fit()` (e.g., the `max_depth` of a Decision Tree, or the `learning_rate` of a Neural Network).

### 2. Grid Search (The Exhaustive Search)
You provide a list of values for each hyperparameter. The algorithm tries *every single possible combination*.
- **Pros**: Guaranteed to find the best combination out of the options you provided.
- **Cons**: Extremely slow. If you have 5 knobs with 5 settings each, that's $5^5 = 3125$ models to train!

### 3. Random Search (The Efficient Search)
You define a range for each hyperparameter. The algorithm randomly picks combinations for a set number of iterations (e.g., 100 times).
- **Pros**: Much faster. Surprisingly, it often finds a combination that is 99% as good as Grid Search in a fraction of the time.
- **Cons**: Not guaranteed to find the *absolute* mathematical best combination.

### 4. The Validation Set (Why we use CV)
You cannot tune your model using the Test Set. If you do, you are just repeatedly tweaking the model until it memorizes the Test Set! Therefore, tuning algorithms use **Cross-Validation** on the Training set to figure out the best settings.

## Output / Result Interpretation
The output of a tuning algorithm is simply the "best" model it found. You can then look at `best_params_` to see exactly which settings won the competition.

## Real-World Uses
- **Kaggle Competitions**: The difference between 1st place and 100th place is almost always who did a better job of hyperparameter tuning.
- **Deploying to Production**: Before deploying a final model to serve customers, it is standard practice to run an overnight Grid Search to squeeze out every last drop of performance.

## Common Mistakes
- **Tuning on the Test Set**: This completely ruins the integrity of your test set (Data Leakage).
- **Huge Grids**: Giving a Grid Search too many options and crashing your computer. Always start with a small Random Search to find the right "neighborhood", then do a tight Grid Search.
- **Ignoring the Defaults**: Sometimes, the default settings actually *are* the best. Always check if your tuned model actually beats the baseline model.

## Related Methods
- **Bayesian Optimization**: An advanced technique (using libraries like `Optuna` or `Hyperopt`) that learns from past attempts. If a certain combination was terrible, it avoids that "area" of settings on the next try.

## Code References
- `code/example-01-grid-search.py`
- `code/example-02-random-search.py`
- `code/example-03-nested-cv.py`


---

## Method & Options: Hyperparameter Tuning

This document details the common scikit-learn functions used to tune models. They are found in `sklearn.model_selection`.

### 1. `GridSearchCV`

#### Purpose
Performs an exhaustive search over a specified parameter grid, using cross-validation to evaluate each combination.

#### Syntax
```python
from sklearn.model_selection import GridSearchCV

# Define the dictionary of settings to try
param_grid = {
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5]
}

# Initialize the searcher
grid_search = GridSearchCV(
    estimator=model, 
    param_grid=param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1 
)

# Run the search (This takes time!)
grid_search.fit(X_train, y_train)
```

#### Common Arguments
- `estimator`: The untrained machine learning model.
- `param_grid` (dict): Dictionary with parameters names (string) as keys and lists of parameter settings to try as values.
- `cv` (int): Number of Cross-Validation folds. (e.g., `5` means every combination is trained 5 times).
- `scoring` (string): The metric to optimize for (e.g., `'f1'`, `'neg_mean_squared_error'`).
- `n_jobs` (int): Number of CPU cores to use. Set to `-1` to use all available cores (highly recommended to speed it up).

#### Important Attributes (After fitting)
- `grid_search.best_params_`: A dictionary showing the winning combination of settings.
- `grid_search.best_estimator_`: The actual trained model using the winning settings. You can immediately call `.predict()` on this.
- `grid_search.best_score_`: The cross-validation score the winning model achieved.

---

### 2. `RandomizedSearchCV`

#### Purpose
Performs a randomized search over hyperparameters. Instead of trying every combination, it samples randomly from distributions.

#### Syntax
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Define distributions instead of lists
param_dist = {
    'max_depth': randint(3, 20),      # Random integers between 3 and 20
    'min_samples_split': randint(2, 10)
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=20,     # Try exactly 20 random combinations
    cv=5,
    random_state=42
)

random_search.fit(X_train, y_train)
```

#### Common Arguments
- `param_distributions` (dict): Dictionary with parameter names as keys and distributions (like `scipy.stats.randint`) or lists as values.
- `n_iter` (int): Number of parameter settings that are sampled. Trades off runtime vs quality of the solution.

#### Workflow Best Practice
1. Use `RandomizedSearchCV` first with a wide range of values and `n_iter=50` to find roughly where the best parameters live.
2. Look at the `best_params_`.
3. Create a tight `param_grid` around those specific values and run `GridSearchCV` to find the exact mathematical peak.

---

## Hyperparameter Tuning Examples

This document explains the python examples provided in the `code/` directory.

### 1. Grid Search (`example-01-grid-search.py`)
This script demonstrates the "brute force" method of finding the best model.
- It loads a dataset and initializes a Random Forest model.
- It defines a `param_grid` with a few different settings for the forest.
- It uses `GridSearchCV` to try every single combination, using cross-validation to ensure the results are robust.
- It extracts the `best_estimator_` and shows how much better it performs compared to a default model.

### 2. Random Search (`example-02-random-search.py`)
This script demonstrates the "smart and fast" method.
- It uses `scipy.stats` to define *ranges* of numbers (e.g., any number between 10 and 1000) rather than a hardcoded list.
- It uses `RandomizedSearchCV` to try exactly 20 random combinations.
- It demonstrates that Random Search usually finds a near-perfect model much faster than an exhaustive Grid Search would take on the same ranges.

### 3. Nested Cross-Validation (`example-03-nested-cv.py`)
This is an advanced concept for when you want to report the absolute most unbiased performance estimate possible.
- Standard Grid Search uses CV to *tune* the model, but it still evaluates the final model on a single static test set.
- Nested CV puts a Grid Search *inside* another Cross-Validation loop.
- It prevents "overfitting to the validation set" and provides a highly conservative estimate of how the model will perform in the wild.

---

## Practice Exercises: Hyperparameter Tuning

### Exercise 1: Exploring Parameters
1. Load the `wine` dataset from `sklearn.datasets`.
2. Do a train-test split.
3. Train a standard `RandomForestClassifier()` with no arguments.
4. Calculate the accuracy on the test set.
5. Train a second `RandomForestClassifier(max_depth=2, n_estimators=10)`.
6. Calculate its accuracy. Did it go up or down? By tweaking these numbers, you are manually hyperparameter tuning!

### Exercise 2: Implementing Grid Search
1. Using the data from Exercise 1, define a parameter grid:
   `param_grid = {'max_depth': [2, 5, None], 'n_estimators': [10, 50, 100], 'criterion': ['gini', 'entropy']}`
2. How many total combinations does this grid have?
3. Initialize `GridSearchCV(model, param_grid, cv=5)`.
4. Fit the Grid Search on the *Training* data.
5. Print `grid_search.best_params_`. What was the winning combination?
6. Print `grid_search.score(X_test, y_test)` to see how the winning model performs on the holdout test set.

### Exercise 3: Random Search vs Grid Search
1. Create a massive parameter grid for a Random Forest:
   `param_dist = {'max_depth': range(1, 50), 'n_estimators': range(10, 500)}`
2. If you used Grid Search on this, how many combinations would it try? (Hint: $49 \times 490 = 24,010$). Assuming 5-fold CV, that's over 120,000 model trainings! It would take a long time.
3. Instead, use `RandomizedSearchCV` with `n_iter=20`.
4. Fit it to the training data.
5. Did it find a model with a good test score in a fraction of the time?

### Exercise 4: Information Leakage Check
You decide to fill missing values in your dataset with the mean. 
- **Method A**: You fill the missing values using the mean of the entire dataset. Then you use `GridSearchCV`.
- **Method B**: You build a Scikit-Learn `Pipeline` that fills missing values and trains the model. You pass the *Pipeline* into `GridSearchCV`.
**Question**: Why is Method A considered Data Leakage, and why is Method B the correct way to do it? (Hint: Think about what the Validation folds inside the Grid Search are seeing).

---

## Interview Questions: Hyperparameter Tuning

### Beginner Questions
1. **What is the difference between a Parameter and a Hyperparameter?**
   - *Answer*: A parameter is learned automatically by the model from the data (like the weights in a regression equation). A hyperparameter is set manually by the data scientist *before* training begins (like the maximum depth of a tree).
2. **What does a Grid Search actually do?**
   - *Answer*: It takes a dictionary of different hyperparameter values, generates every single possible combination of those values, and trains a model for each combination to see which one performs the best.

### Conceptual Questions
3. **Why do we need a Validation set (or Cross-Validation) during hyperparameter tuning? Why not just use the Test set?**
   - *Answer*: If you tune your hyperparameters to get the highest score on the Test set, you are causing "Data Leakage." You are essentially manually tweaking the model until it memorizes the Test data. The Test set must remain completely hidden until the very end. Therefore, we use a separate Validation set (or CV on the training data) to pick the best hyperparameters.
4. **When would you choose Random Search over Grid Search?**
   - *Answer*: When the hyperparameter space is very large or continuous. Grid Search on 5 hyperparameters with 10 options each requires 100,000 model trainings. Random Search can explore the same space and often find a near-optimal solution in just 100 or 200 iterations, saving immense amounts of time.
5. **If I run Grid Search and the winning hyperparameters are at the absolute edge of my grid (e.g., I searched `max_depth` [2, 4, 6] and 6 won), what should I do?**
   - *Answer*: You should expand your grid in that direction! If 6 won, the true optimal value might be 8 or 10. You should run another search checking [6, 8, 10, 12].

### Practical / Scenario Questions
6. **You run a Grid Search with `cv=5` on a grid with 100 combinations. How many times is the `fit()` method called in total?**
   - *Answer*: 500 times. (100 combinations * 5 folds). This is why tuning can be computationally expensive.
7. **Your baseline model got 85% accuracy. You spent 3 days running a massive Grid Search, and the tuned model gets 85.1% accuracy. Should you deploy the tuned model?**
   - *Answer*: Usually, no. The principle of Occam's Razor (or the "Simplest Solution") applies. If tuning only provides a microscopic benefit, it is usually better to deploy the simpler, baseline model, as the heavily tuned model might be slightly overfitting to the training data.

---

## Python Code Examples

### `example-01-grid-search.py`

```python
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
```

### `example-02-random-search.py`

```python
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
```

### `example-03-nested-cv.py`

```python
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
```
