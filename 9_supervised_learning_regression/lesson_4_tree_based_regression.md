# Tree-Based Regression

## Learning Objective
Students will learn how Decision Trees, Random Forests, and Gradient Boosting algorithms (like XGBoost) partition data to make highly accurate non-linear predictions without requiring extensive feature scaling.

## What Is This Topic?
Tree-based models make predictions by asking a series of "Yes/No" questions about the data. A Decision Tree is a single flowchart. A Random Forest is a crowd of trees voting on the answer. Gradient Boosting is a team of trees where each new tree tries to fix the mistakes of the previous one.

## Why This Topic Matters
In industry, ensemble tree-based models (Random Forest, XGBoost, LightGBM) are the absolute "go-to" algorithms for tabular data (data in spreadsheets or SQL tables). They often win Kaggle competitions and provide the best out-of-the-box accuracy with minimal data cleaning.

## Core Intuition
- **Decision Tree**: "Is the house size > 2000 sq ft? Yes. Is it located in NY? No." At the end of the branches, it predicts the average price of all training houses that fit those exact criteria.
- **Random Forest**: Builds 100 different Decision Trees on random subsets of the data and random subsets of features. The final prediction is the average of all 100 trees' predictions. This stops the model from overfitting.
- **Gradient Boosting (XGBoost)**: Builds Tree 1. Tree 1 gets some predictions wrong. Builds Tree 2 specifically to predict the *errors* of Tree 1. Tree 3 predicts the errors of Tree 1+2. They work together sequentially.

## Key Concepts
- **Splitting Criterion (Variance Reduction/MSE)**: How a tree decides where to split a node. It tries to group data so that points in the resulting groups have similar target values (low variance).
- **Max Depth**: How many layers of questions the tree can ask. Deep trees overfit easily.
- **Bagging (Bootstrap Aggregating)**: The technique used by Random Forest (building multiple independent models in parallel).
- **Boosting**: Building models sequentially to correct past errors.
- **Feature Importance**: Trees can naturally calculate which features were used the most to make splits, making the model highly interpretable at a macro level.

## Advantages
- Does **not** require feature scaling (standardization/normalization).
- Handles non-linear relationships naturally.
- Handles outliers and missing values well (especially XGBoost).
- Provides Feature Importance scores.

## Limitations
- **Decision Trees**: Highly prone to overfitting.
- **Random Forests**: Can be slow to predict if the forest is huge.
- **Gradient Boosting**: Prone to overfitting if not tuned carefully; harder to tune than Random Forests.
- Tree models **cannot extrapolate**. If you train a tree on houses priced between $100k-$500k, it can *never* predict a price of $600k for new data, unlike linear regression.

## Code References
- `code/example-01-decision-tree.py` — basic tree intuition
- `code/example-02-random-forest.py` — the workhorse model
- `code/example-03-xgboost-real-world.py` — the industry standard


---

## Method Options: Tree-Based Regression

### `sklearn.tree.DecisionTreeRegressor`

**Purpose**: A single decision tree for regression. Mostly used for teaching or as base estimators for ensembles.

**Syntax**:
```python
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=3)
```

**Common Arguments**:
- `max_depth` (int, default `None`): The maximum depth of the tree. Crucial for preventing overfitting.
- `min_samples_split` (int, default `2`): The minimum number of samples required to split an internal node.
- `random_state` (int): Seed for reproducible splits.

---

### `sklearn.ensemble.RandomForestRegressor`

**Purpose**: An ensemble of decision trees using bagging. Highly robust and rarely overfits.

**Syntax**:
```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, max_depth=None)
```

**Common Arguments**:
- `n_estimators` (int, default `100`): The number of trees in the forest. More is generally better but takes longer to compute.
- `max_depth` (int, default `None`): Maximum depth of individual trees.
- `max_features` (str or int, default `1.0`): Number of features to consider when looking for the best split. Essential for ensuring the trees are diverse.
- `n_jobs` (int, default `None`): Set to `-1` to use all CPU cores.

**Common Attributes**:
- `feature_importances_`: Array indicating the importance of each feature.

---

### `xgboost.XGBRegressor`

**Purpose**: Optimized distributed gradient boosting library. The standard for high-performance tabular modeling.

**Note**: Not part of `sklearn`. Must be installed via `pip install xgboost`.

**Syntax**:
```python
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=100, learning_rate=0.1)
```

**Common Arguments**:
- `n_estimators` (int, default `100`): Number of boosting rounds (trees).
- `learning_rate` / `eta` (float, default `0.3`): Step size shrinkage used to prevent overfitting. Usually set between 0.01 and 0.1.
- `max_depth` (int, default `6`): Maximum depth of a tree.
- `subsample` (float, default `1.0`): Subsample ratio of the training instances. Setting it to 0.5 means XGBoost randomly samples half the training data prior to growing trees.
- `colsample_bytree` (float, default `1.0`): Subsample ratio of columns when constructing each tree.

**Workflow**:
```python
model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)
# Early stopping can be used via fit parameters
model.fit(X_train, y_train, 
          eval_set=[(X_test, y_test)], 
          early_stopping_rounds=10, 
          verbose=False)
```

---

## Practice: Tree-Based Regression

### Conceptual Questions
1. Why does a Decision Tree Regressor not require feature scaling like StandardScaling, whereas Linear Regression with Ridge/Lasso does?
2. Explain the difference in how Random Forest and Gradient Boosting handle errors made by trees.
3. If your Random Forest model has a massive gap between Training R2 (0.98) and Test R2 (0.60), what parameter would you adjust first and in what direction?

### Coding Tasks

#### Task 1: Overfitting a Decision Tree
1. Load a regression dataset (e.g., California Housing).
2. Train a `DecisionTreeRegressor` with `max_depth=None` (the default).
3. Evaluate the R2 score on the Training Set and the Test Set. (Notice the training set score is likely 1.0 or very close to it).
4. Retrain the model using `max_depth=5`. Compare the new Train/Test scores.

#### Task 2: Random Forest Feature Importance
1. Load a dataset with many features.
2. Train a `RandomForestRegressor(n_estimators=100)`.
3. Extract `model.feature_importances_`.
4. Create a Pandas DataFrame mapping feature names to their importance scores.
5. Sort the DataFrame in descending order and use Matplotlib/Seaborn to create a horizontal bar chart of the top 10 most important features.

#### Task 3: Introduction to XGBoost
1. Install `xgboost` if not already installed.
2. Train an `XGBRegressor` on a dataset.
3. Experiment with the `learning_rate` parameter (try 0.01, 0.1, and 0.5). How does it affect the final test score?

---

## Interview Questions: Tree-Based Regression

### Beginner Level
1. **How does a Decision Tree make a prediction for a continuous target variable?**
   - *Expected Answer*: It traverses the tree by answering the feature splits (yes/no questions) until it reaches a leaf node. The prediction is the average (mean) of all the training target values that fell into that specific leaf node.

2. **What is a Random Forest?**
   - *Expected Answer*: It is an ensemble method that builds many decision trees and averages their predictions. To ensure the trees are diverse, it trains each tree on a random sample of the data (bootstrapping) and looks at a random subset of features at each split.

### Intermediate Level
3. **What is the difference between Bagging and Boosting?**
   - *Expected Answer*: Bagging (like Random Forest) builds multiple independent models in parallel and averages them to reduce variance. Boosting (like XGBoost) builds models sequentially, where each new model tries to correct the errors (residuals) of the combined previous models, reducing both bias and variance.

4. **Why are Tree-based models robust to outliers?**
   - *Expected Answer*: Because they split data based on thresholds (e.g., "is income > $100k?"). Whether an income is $105k or $10 million, it falls into the same bucket and doesn't pull a "line of best fit" toward it like it would in linear regression.

5. **Can a Random Forest Regressor extrapolate? (Predict a value higher than any value in its training set?)**
   - *Expected Answer*: No. Because the prediction at a leaf node is the average of the training samples in that node, it is mathematically impossible for a tree to predict a value higher than the maximum (or lower than the minimum) value seen in the training data.

### Advanced / Practical Level
6. **If your Gradient Boosting model (XGBoost) is overfitting, what parameters would you tune?**
   - *Expected Answer*: I would lower the `learning_rate` (and proportionally increase `n_estimators`), decrease `max_depth` to make individual trees shallower, or use subsampling parameters like `subsample` or `colsample_bytree` to introduce more randomness.

7. **How is Feature Importance calculated in a Random Forest?**
   - *Expected Answer*: It is usually calculated based on the "Gini importance" or "Mean Decrease in Impurity." Every time a feature is used to split a node, it calculates how much that split reduced the variance (or impurity) compared to the parent node. The total variance reduction across all trees for that feature is averaged to get the final importance score.

---

## Python Code Examples

### `example-01-decision-tree.py`

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

# 1. Generate Simple Data (Experience vs Salary)
np.random.seed(42)
experience = np.sort(np.random.uniform(1, 10, 20)).reshape(-1, 1)
# Salary generally goes up with experience, but in steps
salary = np.where(experience < 3, 40000, 
                  np.where(experience < 7, 70000, 100000)).ravel()
salary = salary + np.random.normal(0, 5000, 20) # add noise

# 2. Train a Decision Tree
# A depth of 2 means it can only ask 2 levels of questions
tree_model = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_model.fit(experience, salary)

# 3. Predict to visualize the "step" function
X_plot = np.arange(0, 11, 0.1).reshape(-1, 1)
y_plot = tree_model.predict(X_plot)

# 4. Visualize the Predictions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(experience, salary, color='black', label='Data')
plt.plot(X_plot, y_plot, color='red', lw=2, label='Tree Prediction')
plt.title('Decision Tree: Step Function Predictions')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()

# 5. Visualize the Actual Tree Logic
plt.subplot(1, 2, 2)
plot_tree(tree_model, feature_names=['Experience'], filled=True, rounded=True)
plt.title('The Decision Tree Rules')

plt.tight_layout()
plt.show()

print("Notice how the Decision Tree doesn't draw a smooth line.")
print("Instead, it breaks the data into chunks (e.g., Exp < 2.5) and predicts the average salary for that chunk.")
```

### `example-02-random-forest.py`

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

# 1. Load a real dataset (California Housing)
print("Loading California Housing Data...")
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = california.target # Target is median house value in 100,000s

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Random Forest
# Notice: WE ARE NOT SCALING THE DATA. Trees don't care about scale!
print("Training Random Forest (this might take a few seconds)...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 4. Evaluate
y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("\n--- Model Evaluation ---")
print(f"R-squared: {r2:.4f}")
print(f"RMSE: ${rmse * 100000:.2f}")

# 5. Feature Importance Extraction
print("\n--- Feature Importances ---")
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=True)

# 6. Visualize Feature Importances
plt.figure(figsize=(10, 6))
importances_sorted.plot(kind='barh', color='teal')
plt.title('Random Forest - Feature Importance (California Housing)')
plt.xlabel('Importance Score (Variance Reduction)')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

print("\nInterpretation: 'MedInc' (Median Income in the area) is by far the most important feature for predicting house prices in this model.")
```

### `example-03-xgboost-real-world.py`

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
import xgboost as xgb
import matplotlib.pyplot as plt

# 1. Load Data
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = california.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train XGBoost Regressor
# XGBoost has many hyperparameters. Here are the most common ones.
print("Training XGBoost Model...")
xgb_model = xgb.XGBRegressor(
    n_estimators=300,       # Number of trees
    learning_rate=0.05,     # Step size shrinkage
    max_depth=6,            # Maximum depth of each tree
    subsample=0.8,          # Use 80% of data per tree (prevents overfitting)
    colsample_bytree=0.8,   # Use 80% of features per tree
    random_state=42,
    n_jobs=-1
)

# We can use early stopping to halt training if the validation score stops improving
# Note: eval_set is used as the validation set during training
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50 # Print progress every 50 trees
)

# 3. Evaluate the Final Model
y_pred = xgb_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("\n--- XGBoost Model Evaluation ---")
print(f"R-squared: {r2:.4f}")
print(f"RMSE: ${rmse * 100000:.2f}")

# 4. Plot Training History (Learning Curve)
# Extract the evaluation results
results = xgb_model.evals_result()
val_rmse = results['validation_0']['rmse']

plt.figure(figsize=(10, 5))
plt.plot(val_rmse, label='Validation RMSE')
plt.title('XGBoost Learning Curve')
plt.xlabel('Number of Trees (Boosting Rounds)')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\nNotice how the error drops quickly at first, then starts to plateau.")
print("If we trained for 10,000 trees, the model would likely overfit, and validation RMSE would start rising again.")
```
