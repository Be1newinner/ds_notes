# Tree-Based Classification: Decision Trees & Ensembles

## Learning Objective
Understand how Decision Trees mimic human decision-making, why a single tree is prone to overfitting, and how ensemble methods (Random Forest and Gradient Boosting) combine many weak trees to create state-of-the-art models for tabular data.

## Part 1: Decision Trees

### What Is This Topic?
A Decision Tree classifies data by asking a series of True/False questions about the features. It splits the data into smaller and smaller subsets until it reaches a final decision (a "leaf").

### Core Intuition
Think of a game of "20 Questions." To guess an animal, you ask: "Does it have four legs?" (Yes/No). If Yes, "Does it bark?" (Yes/No). A Decision Tree does exactly this mathematically. It chooses the question that best separates the classes at each step.

### Key Concepts
- **Root Node**: The very first question asked (the top of the tree).
- **Leaf Node**: The final prediction (the bottom of the tree).
- **Gini Impurity / Entropy**: The mathematical metrics used to decide which question to ask. The algorithm wants to split the data so that the resulting groups are as "pure" as possible (e.g., all 1s in one group, all 0s in the other).

### Advantages & Limitations
- **Advantages**: Incredibly interpretable; requires *zero* feature scaling; handles both categorical and numerical data naturally.
- **Limitations**: A single tree is notorious for **overfitting**. It will keep growing until it memorizes every single row in the training data, leading to a massive, complex tree that fails on test data.

---

## Part 2: Ensembles (Random Forest)

### What Is This Topic?
An ensemble model combines the predictions of multiple machine learning models to produce a single, stronger prediction. Random Forest is an ensemble of Decision Trees.

### Core Intuition
If you ask one person to guess the weight of a cow, they might be wildly wrong. If you ask 1,000 people and take the average, the guess will be incredibly accurate. Random Forest creates 100+ different Decision Trees and lets them "vote" on the final classification.

### Key Concepts
- **Bagging (Bootstrap Aggregating)**: Each tree in the forest is trained on a random *sample* of the training data (with replacement).
- **Random Subspace**: At each split in a tree, only a random *subset* of features is considered. 
- **Result**: Because of this randomness, every tree is slightly different. When they vote together, the variance (overfitting) is drastically reduced.

---

## Part 3: Ensembles (Gradient Boosting)

### What Is This Topic?
Gradient Boosting is another way to ensemble trees, but instead of building them all at once (like Random Forest), it builds them sequentially.

### Core Intuition
1. Build a short, weak tree. It makes some mistakes.
2. Build a second tree *specifically designed to fix the mistakes of the first tree*.
3. Build a third tree to fix the mistakes of the second tree.
4. Repeat 100+ times.

### Key Concepts
- **Boosting**: Combining weak learners sequentially to minimize errors.
- **Learning Rate**: Controls how much each tree is allowed to contribute to the final answer. A small learning rate requires more trees but usually results in a better, more robust model.

## Real-World Uses
- **Random Forest**: The ultimate "baseline" model for tabular data. Often used for feature selection because it can rank the importance of features.
- **Gradient Boosting (XGBoost, LightGBM, CatBoost)**: The algorithm that wins almost all Kaggle competitions for structured/tabular data (credit scoring, pricing, etc.).

## Code References
- `code/example-01-decision-tree.py`
- `code/example-02-random-forest.py`
- `code/example-03-gradient-boosting.py`
- `code/example-04-feature-importance.py`


---

## Method Options: Tree-Based Methods in Scikit-Learn

This document explains the primary tools used to implement tree-based algorithms.

### 1. `sklearn.tree.DecisionTreeClassifier`

#### Syntax
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, criterion='gini')
```

#### Common Arguments
- **`criterion`** (`{'gini', 'entropy'}`, default=`'gini'`): The function to measure the quality of a split. They perform very similarly; `gini` is slightly faster to compute.
- **`max_depth`** (`int`, default=`None`): The maximum depth of the tree. If `None`, nodes are expanded until all leaves are pure. **Crucial for preventing overfitting.**
- **`min_samples_split`** (`int`, default=`2`): The minimum number of samples required to split an internal node.
- **`min_samples_leaf`** (`int`, default=`1`): The minimum number of samples required to be at a leaf node.

---

### 2. `sklearn.ensemble.RandomForestClassifier`

#### Syntax
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
```

#### Common Arguments
- **`n_estimators`** (`int`, default=`100`): The number of trees in the forest. More is generally better, but eventually, returns diminish and it just takes longer to train.
- **`max_depth`** (`int`, default=`None`): Unlike a single tree, you often don't need to restrict the depth as much in a Random Forest because the ensemble voting naturally prevents overfitting.
- **`max_features`** (`{'sqrt', 'log2', None}`, default=`'sqrt'`): The number of features to consider when looking for the best split. This introduces the necessary randomness.
- **`n_jobs`** (`int`, default=`None`): Set to `-1` to use all available CPU cores, which drastically speeds up training since trees can be built in parallel.

#### Common Attributes
- **`feature_importances_`**: An array showing how much each feature contributed to decreasing the impurity across all trees. Extremely useful for business insights.

---

### 3. `sklearn.ensemble.GradientBoostingClassifier`

#### Syntax
```python
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
```

#### Common Arguments
- **`n_estimators`** (`int`, default=`100`): The number of boosting stages to perform.
- **`learning_rate`** (`float`, default=`0.1`): Shrinks the contribution of each tree by `learning_rate`. There is a trade-off between `learning_rate` and `n_estimators`. Lower learning rates require more trees.
- **`max_depth`** (`int`, default=`3`): Maximum depth of the individual regression estimators. Boosting works best with very shallow trees (often called "decision stumps").

#### Typical Workflow & Best Practices
- **Scaling is NOT required** for any tree-based models. A tree just makes a split like "Age > 30". It doesn't care if Age is on a scale of 0-100 and Income is on a scale of 0-1,000,000.
- **Missing Values**: While algorithms like XGBoost handle missing values natively, Scikit-Learn's standard implementations usually require you to impute missing values first.

---

## Examples: Tree-Based Methods

Here is a breakdown of the Python examples provided in the `code/` directory.

### 1. Single Decision Tree (`example-01-decision-tree.py`)
- **Goal:** Understand how a single tree easily overfits and how to visualize its decisions.
- **Dataset:** Synthetic classification dataset.
- **Key Concepts Shown:** 
  - Training an unconstrained `DecisionTreeClassifier`.
  - Seeing the 100% training accuracy vs. poor testing accuracy (classic overfitting).
  - Using `max_depth` to constrain the tree and fix the overfitting.
- **Takeaway:** Never use a single, unconstrained Decision Tree in production.

### 2. Random Forest (`example-02-random-forest.py`)
- **Goal:** Show how ensembling fixes the problems of a single tree.
- **Dataset:** The Wine quality dataset.
- **Key Concepts Shown:** 
  - Training a `RandomForestClassifier`.
  - Understanding the `n_estimators` and `n_jobs` parameters.
- **Takeaway:** Random Forest is robust, powerful, and requires almost no tuning to get a "good enough" baseline result.

### 3. Gradient Boosting (`example-03-gradient-boosting.py`)
- **Goal:** Introduce sequential boosting and the concept of learning rates.
- **Dataset:** The Wine quality dataset (same as above, for comparison).
- **Key Concepts Shown:** 
  - Training a `GradientBoostingClassifier`.
  - Showing how `learning_rate` and `n_estimators` interact.
- **Takeaway:** Boosting can often squeeze out better performance than Random Forest, but is more prone to overfitting if not tuned properly.

### 4. Feature Importance (`example-04-feature-importance.py`)
- **Goal:** Learn how to extract business insights from tree-based models.
- **Dataset:** Simulated Employee Attrition (Churn) dataset.
- **Key Concepts Shown:** 
  - Accessing the `.feature_importances_` attribute.
  - Sorting and plotting the results using `matplotlib`.
- **Takeaway:** Tree models are excellent at telling you *which* features matter most, making them invaluable for exploratory data analysis and feature selection.

---

## Practice Exercises: Tree-Based Classification

These exercises are designed to test your conceptual understanding and coding skills.

### Conceptual Questions
1. Why is feature scaling (like `StandardScaler`) completely unnecessary for Decision Trees and Random Forests?
2. Explain the difference between Bagging (Random Forest) and Boosting (Gradient Boosting).
3. If your Random Forest model is overfitting, what is the best hyperparameter to adjust to fix it? (Hint: Think about the depth of the individual trees).

### Coding Tasks

#### Task 1: Overfitting a Decision Tree
1. Load the `digits` dataset from `sklearn.datasets`.
2. Split into 80/20 train/test.
3. Train a `DecisionTreeClassifier` with absolutely no parameters (default).
4. Print the accuracy on the *training* set, and then the accuracy on the *test* set. What do you observe?
5. Train a new tree, but set `max_depth=5`. Compare the train and test accuracies again.

#### Task 2: The Power of the Forest
Using the same `digits` dataset:
1. Train a `RandomForestClassifier` with `n_estimators=10` and print the test accuracy.
2. Train another with `n_estimators=100` and print the test accuracy.
3. Did the accuracy improve? Did it take noticeably longer to train?

#### Task 3: Extracting Feature Importance
1. Load the `breast_cancer` dataset (`load_breast_cancer`).
2. Train a `RandomForestClassifier`.
3. Extract the `feature_importances_`.
4. Write a script to find and print the name of the **single most important feature** in determining whether a tumor is malignant or benign according to the model.

---

## Interview Questions: Tree-Based Methods

### Beginner Questions
1. **Explain how a Decision Tree works to a non-technical person.**
   *Hint:* It's like playing a game of 20 questions. It asks a series of True/False questions about the data to narrow down the possibilities until it makes a final guess.
2. **What is a Random Forest?**
   *Hint:* It is an ensemble (collection) of many different decision trees. They all make a prediction, and the forest takes a majority vote to decide the final answer.
3. **Do you need to scale or normalize data before using a Random Forest?**
   *Hint:* No. Trees only care about order and splitting points (e.g., "Is Salary > 50,000?"). The absolute scale of the number doesn't matter at all.

### Conceptual Questions
4. **Why is a single Decision Tree almost never used in production?**
   *Hint:* It is highly prone to overfitting. If allowed to grow deep enough, it will perfectly memorize the training data and fail miserably on new, unseen data.
5. **How does a Random Forest ensure that its trees are actually different from one another?**
   *Hint:* Through two types of randomness: 
   1) **Row sampling (Bagging):** Each tree gets a random sample of the training rows. 
   2) **Feature sampling:** At every single split, the tree is only allowed to choose from a random subset of the features (usually the square root of the total number of features).
6. **What is the fundamental difference in how Random Forests and Gradient Boosting Machines (GBMs) build their trees?**
   *Hint:* Random Forests build trees *independently and in parallel* (Bagging). GBMs build trees *sequentially*, where each new tree tries to correct the errors made by the previous trees (Boosting).

### Practical Questions
7. **If your Random Forest is overfitting, what hyperparameters should you tune?**
   *Hint:* Decrease `max_depth` to make the trees shallower. Increase `min_samples_split` or `min_samples_leaf` to prevent nodes from splitting on very few data points. 
8. **What does the `learning_rate` parameter do in a Gradient Boosting model?**
   *Hint:* It scales the contribution of each newly added tree. A lower learning rate means each tree contributes less to the final prediction, making the model more robust and less prone to overfitting, but it will require more trees (`n_estimators`) to train.
9. **How do tree-based models calculate Feature Importance?**
   *Hint:* They look at every node where a specific feature was used to make a split, and calculate how much that split decreased the "impurity" (Gini or Entropy). The more a feature decreases impurity across all trees in the forest, the more important it is.

### Output Interpretation
10. **Your Random Forest model has a feature importance score of 0.00 for the feature "Customer Age". What does this mean?**
    *Hint:* It means that across all 100+ trees in the forest, "Customer Age" was never used to make a split, or if it was, it didn't improve the purity of the classification at all. You could likely drop this feature without hurting model performance.

---

## Python Code Examples

### `example-01-decision-tree.py`

```python
"""
Example 01: The Overfitting Problem of a Single Decision Tree
Goal: Prove that an unconstrained Decision Tree will overfit, and show how to fix it.
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Create a slightly complex synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- EXPERIMENT 1: Unconstrained Tree ---
# We do not set max_depth. The tree will grow until every leaf is pure.
tree_overfit = DecisionTreeClassifier(random_state=42)
tree_overfit.fit(X_train, y_train)

print("--- Unconstrained Decision Tree ---")
print(f"Training Accuracy: {accuracy_score(y_train, tree_overfit.predict(X_train)) * 100}%")
print(f"Testing Accuracy:  {accuracy_score(y_test, tree_overfit.predict(X_test)) * 100:.2f}%")
print(f"Depth of the tree: {tree_overfit.get_depth()}")
print("Notice the 100% training accuracy. The tree memorized the training data, leading to worse test accuracy.\n")

# --- EXPERIMENT 2: Constrained Tree (Pruning) ---
# We limit the depth to 4.
tree_pruned = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_pruned.fit(X_train, y_train)

print("--- Constrained Decision Tree (max_depth=4) ---")
print(f"Training Accuracy: {accuracy_score(y_train, tree_pruned.predict(X_train)) * 100:.2f}%")
print(f"Testing Accuracy:  {accuracy_score(y_test, tree_pruned.predict(X_test)) * 100:.2f}%")
print("By stopping the tree from growing too deep, it generalized better to the unseen test data.")
```

### `example-02-random-forest.py`

```python
"""
Example 02: Random Forest Classification
Goal: Use the industry-standard Random Forest to improve upon a single tree's performance.
"""

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the Wine Quality dataset (Predicting 3 different classes of wine)
wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Initialize Random Forest
# n_estimators = 100 means we are building 100 different decision trees.
# n_jobs = -1 tells the computer to build these trees in parallel using all CPU cores.
rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

# 3. Train the model
# NOTE: We did NOT scale the data. Tree models do not require scaling!
rf_model.fit(X_train, y_train)

# 4. Evaluate
y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Random Forest Accuracy: {acc * 100:.2f}%\n")
print("--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

print("Random Forest almost always beats a single decision tree out-of-the-box.")
```

### `example-03-gradient-boosting.py`

```python
"""
Example 03: Gradient Boosting
Goal: Implement a sequential ensemble method and understand the learning rate.
"""

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# 1. Load Data (same as Random Forest example for comparison)
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=42)

# 2. Experiment with different Learning Rates
learning_rates = [1.0, 0.1, 0.01]

print("--- Gradient Boosting with Different Learning Rates ---")
for lr in learning_rates:
    # Initialize Gradient Boosting
    # max_depth=3 (Boosting works best with very shallow trees)
    gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=lr, max_depth=3, random_state=42)
    
    # Train
    gbm.fit(X_train, y_train)
    
    # Predict & Evaluate
    train_acc = accuracy_score(y_train, gbm.predict(X_train))
    test_acc = accuracy_score(y_test, gbm.predict(X_test))
    
    print(f"Learning Rate: {lr:5} | Train Acc: {train_acc*100:6.2f}% | Test Acc: {test_acc*100:6.2f}%")

print("\nTakeaway:")
print("A learning rate of 1.0 (too fast) overfits the data.")
print("A learning rate of 0.1 provides a good balance.")
print("A learning rate of 0.01 (too slow) underfits because 100 trees aren't enough at that speed.")
```

### `example-04-feature-importance.py`

```python
"""
Example 04: Feature Importance
Goal: Learn how to extract business insights by identifying which features drive the model's decisions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Simulate Employee Attrition (Churn) Data
np.random.seed(42)
n_employees = 1000

data = {
    'Salary': np.random.normal(60000, 15000, n_employees),
    'Years_at_Company': np.random.randint(1, 10, n_employees),
    'Distance_to_Work': np.random.randint(1, 30, n_employees),
    'Performance_Score': np.random.randint(1, 6, n_employees),
    'Number_of_Projects': np.random.randint(1, 8, n_employees)
}
df = pd.DataFrame(data)

# Create a complex logic for why someone quits (1 = Quit, 0 = Stayed)
# Low salary AND long distance OR terrible performance
quit_prob = np.where((df['Salary'] < 50000) & (df['Distance_to_Work'] > 20), 0.8, 0.1)
quit_prob = np.where(df['Performance_Score'] == 1, 0.9, quit_prob)

df['Quit'] = (np.random.rand(n_employees) < quit_prob).astype(int)

X = df.drop('Quit', axis=1)
y = df['Quit']

# 2. Train Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Extract Feature Importances
importances = model.feature_importances_
feature_names = X.columns

# 4. Create a DataFrame to make it look nice
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort from highest to lowest
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("--- Feature Importance ---")
print(importance_df.to_string(index=False))

print("\nBusiness Insight:")
print("The model clearly identified that Performance Score, Salary, and Distance to Work")
print("are the primary drivers of employee attrition. 'Number of Projects' has almost no impact.")

# Note: In a real environment, you would usually plot this using matplotlib or seaborn.
```
