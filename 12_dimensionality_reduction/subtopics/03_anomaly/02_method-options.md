# Isolation Forest Method and Options

## Scikit-Learn: `sklearn.ensemble.IsolationForest`

### Purpose
To isolate observations by randomly selecting a feature and then randomly selecting a split value. The number of splittings required to isolate a sample is equivalent to the path length from the root node to the terminating node. Anomalies have shorter paths.

### Syntax
```python
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
```

### Common Arguments
- `n_estimators` (int): Number of trees (default=100).
- `contamination` (float or 'auto'): The proportion of outliers in the dataset. Defines the threshold on the decision function.
- `random_state` (int): Reproducibility.

### Common Methods
- `fit(X)`: Fit the model.
- `predict(X)`: Predict if a particular sample is an outlier or not. Returns `1` for inliers, `-1` for outliers.
- `decision_function(X)`: Average anomaly score. The lower, the more abnormal.

### Typical Workflow
1. **Preprocess**: Scale your features (e.g., using `StandardScaler`) if features have vastly different ranges.
2. **Initialize**: Create `IsolationForest` with an estimated `contamination` rate.
3. **Fit**: Call `fit()` on your dataset.
4. **Predict**: Call `predict()` to get the labels (`1` or `-1`).
5. **Filter**: Examine the rows flagged as `-1` to understand why they were flagged.

### Common Mistakes
- **Confusing output labels**: Remember that `1` is GOOD/NORMAL, and `-1` is BAD/ANOMALY. This often trips up beginners.
