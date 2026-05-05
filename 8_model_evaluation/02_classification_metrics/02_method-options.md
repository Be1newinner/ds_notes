# Method & Options: Classification Metrics

This document details the common scikit-learn functions used to evaluate classification models. All these functions are found in `sklearn.metrics`.

## 1. `confusion_matrix`

### Purpose
Calculates the raw counts of True Negatives, False Positives, False Negatives, and True Positives.

### Syntax
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
```

### Output Interpretation
Returns a 2D array (list of lists). For binary classification:
`[[TN, FP],`
 `[FN, TP]]`
To easily visualize this, you usually pass the `cm` to `sns.heatmap()` or use `ConfusionMatrixDisplay`.

---

## 2. `accuracy_score`, `precision_score`, `recall_score`, `f1_score`

### Purpose
Calculates the specific metric as a single float number between 0.0 and 1.0.

### Syntax
```python
from sklearn.metrics import precision_score, recall_score
p = precision_score(y_true, y_pred)
r = recall_score(y_true, y_pred)
```

### Common Arguments
- `y_true`: The actual ground truth labels.
- `y_pred`: The labels predicted by your model.
- `pos_label` (int/str): Which class should be considered the "Positive" class (default is `1`).
- `average` (string): Vital for multi-class problems (e.g., predicting 3 or more categories).
  - `'binary'`: (Default) Only reports results for the class specified by `pos_label`.
  - `'macro'`: Calculates metrics for each class individually, then takes the unweighted mean. Does not take class imbalance into account.
  - `'weighted'`: Calculates metrics for each class, then takes the average weighted by the number of true instances for each class.

---

## 3. `classification_report`

### Purpose
The ultimate time-saver. It builds a text report showing the main classification metrics (precision, recall, f1, support) for every single class in your dataset.

### Syntax
```python
from sklearn.metrics import classification_report
report = classification_report(y_true, y_pred, target_names=["Not Spam", "Spam"])
print(report)
```

### Output Interpretation
- **support**: The actual number of occurrences of that class in the `y_true` dataset.
- **macro avg**: The standard average of the metrics across all classes.
- **weighted avg**: The average weighted by the `support` (the size of each class).

---

## 4. `roc_auc_score`

### Purpose
Calculates the Area Under the Receiver Operating Characteristic Curve. This metric evaluates how well the model can distinguish between the classes at various threshold levels.

### Syntax
```python
from sklearn.metrics import roc_auc_score
# NOTE: You MUST pass probabilities, not just the final 0/1 predictions!
y_pred_proba = model.predict_proba(X_test)[:, 1] 
auc = roc_auc_score(y_true, y_pred_proba)
```

### Important Rule
Unlike `accuracy` or `precision`, `roc_auc_score` should be fed the *probabilities* outputted by the model (`predict_proba()`), not the hard class predictions (`predict()`). You want the probability of the positive class (usually column index `1`).
