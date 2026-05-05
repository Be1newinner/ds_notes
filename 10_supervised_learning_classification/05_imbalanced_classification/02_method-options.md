# Method Options: Imbalanced Classification

This document covers the `imbalanced-learn` library and `class_weight` parameters.

## 1. Algorithmic Approach: `class_weight`

Most models in Scikit-Learn (LogisticRegression, RandomForestClassifier, SVC, DecisionTreeClassifier) accept a `class_weight` argument.

### Syntax
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight='balanced')
```

### Options
- **`None`** (default): All classes carry weight one.
- **`'balanced'`**: Automatically adjusts weights inversely proportional to class frequencies in the input data: `n_samples / (n_classes * np.bincount(y))`.
- **Custom Dictionary**: You can manually pass a dictionary: `class_weight={0: 1, 1: 10}` means Class 1 is penalized 10 times harder than Class 0 for a mistake.

---

## 2. Data Approach: `imbalanced-learn` (imblearn)

`imblearn` is a library built specifically to work with scikit-learn for handling imbalanced data.

### Oversampling with SMOTE
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

**Common Arguments**:
- **`sampling_strategy`**: 
  - `'auto'` (default): resample all classes but the majority class.
  - `float`: (e.g., `0.5`) means resample the minority class until it is 50% the size of the majority class.
- **`k_neighbors`** (`int`, default=`5`): Number of nearest neighbors to used to construct synthetic samples.

### Undersampling with RandomUnderSampler
```python
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
```

### The `imblearn` Pipeline
Because you should **never** apply SMOTE to your validation/test set, you cannot use Scikit-Learn's standard `Pipeline`. If you cross-validate, SMOTE will leak into the validation folds. You must use `imblearn.pipeline.Pipeline`.

```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('smote', SMOTE()),
    ('model', LogisticRegression())
])
# When you call cross_val_score on this pipeline, it correctly applies 
# SMOTE ONLY to the training folds, leaving the validation fold untouched.
```
