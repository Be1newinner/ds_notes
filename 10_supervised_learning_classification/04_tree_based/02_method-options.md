# Method Options: Tree-Based Methods in Scikit-Learn

This document explains the primary tools used to implement tree-based algorithms.

## 1. `sklearn.tree.DecisionTreeClassifier`

### Syntax
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, criterion='gini')
```

### Common Arguments
- **`criterion`** (`{'gini', 'entropy'}`, default=`'gini'`): The function to measure the quality of a split. They perform very similarly; `gini` is slightly faster to compute.
- **`max_depth`** (`int`, default=`None`): The maximum depth of the tree. If `None`, nodes are expanded until all leaves are pure. **Crucial for preventing overfitting.**
- **`min_samples_split`** (`int`, default=`2`): The minimum number of samples required to split an internal node.
- **`min_samples_leaf`** (`int`, default=`1`): The minimum number of samples required to be at a leaf node.

---

## 2. `sklearn.ensemble.RandomForestClassifier`

### Syntax
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
```

### Common Arguments
- **`n_estimators`** (`int`, default=`100`): The number of trees in the forest. More is generally better, but eventually, returns diminish and it just takes longer to train.
- **`max_depth`** (`int`, default=`None`): Unlike a single tree, you often don't need to restrict the depth as much in a Random Forest because the ensemble voting naturally prevents overfitting.
- **`max_features`** (`{'sqrt', 'log2', None}`, default=`'sqrt'`): The number of features to consider when looking for the best split. This introduces the necessary randomness.
- **`n_jobs`** (`int`, default=`None`): Set to `-1` to use all available CPU cores, which drastically speeds up training since trees can be built in parallel.

### Common Attributes
- **`feature_importances_`**: An array showing how much each feature contributed to decreasing the impurity across all trees. Extremely useful for business insights.

---

## 3. `sklearn.ensemble.GradientBoostingClassifier`

### Syntax
```python
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
```

### Common Arguments
- **`n_estimators`** (`int`, default=`100`): The number of boosting stages to perform.
- **`learning_rate`** (`float`, default=`0.1`): Shrinks the contribution of each tree by `learning_rate`. There is a trade-off between `learning_rate` and `n_estimators`. Lower learning rates require more trees.
- **`max_depth`** (`int`, default=`3`): Maximum depth of the individual regression estimators. Boosting works best with very shallow trees (often called "decision stumps").

### Typical Workflow & Best Practices
- **Scaling is NOT required** for any tree-based models. A tree just makes a split like "Age > 30". It doesn't care if Age is on a scale of 0-100 and Income is on a scale of 0-1,000,000.
- **Missing Values**: While algorithms like XGBoost handle missing values natively, Scikit-Learn's standard implementations usually require you to impute missing values first.
