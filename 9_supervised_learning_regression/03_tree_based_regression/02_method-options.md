# Method Options: Tree-Based Regression

## `sklearn.tree.DecisionTreeRegressor`

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

## `sklearn.ensemble.RandomForestRegressor`

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

## `xgboost.XGBRegressor`

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
