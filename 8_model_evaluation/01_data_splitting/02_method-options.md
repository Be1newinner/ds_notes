# Method & Options: Data Splitting

This document details the common scikit-learn methods used to split and validate datasets.

## 1. `sklearn.model_selection.train_test_split`

### Purpose
The standard, go-to function for making a single random split of your data into training and testing sets.

### Syntax
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Common Arguments
- `arrays` (X, y): The datasets you want to split.
- `test_size` (float): The proportion of the dataset to include in the test split (e.g., `0.2` or `0.3`).
- `train_size` (float): Rarely used if `test_size` is defined.
- `random_state` (int): A seed value. Setting this to a number (like `42`) ensures that you get the exact same split every time you run the code. Crucial for reproducibility.
- `shuffle` (boolean): Default `True`. Whether to shuffle data before splitting.
- `stratify` (array-like): If not None, data is split in a stratified fashion, using this as the class labels. You usually pass `y` here.

### Return Type
Returns a list containing the train-test split of the arrays you passed in (usually unpacked into 4 variables).

---

## 2. `sklearn.model_selection.KFold`

### Purpose
Used to perform Cross-Validation. It divides the dataset into `k` consecutive folds. Each fold is used once as a validation while the `k - 1` remaining folds form the training set.

### Syntax
```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X):
    # Training loop here
```

### Common Arguments
- `n_splits` (int): Number of folds. Must be at least 2. Default is 5.
- `shuffle` (boolean): Whether to shuffle the data before splitting into batches.
- `random_state` (int): Used when `shuffle` is True.

### Workflow
Usually, you don't write the `for` loop yourself. You pass the `KFold` object into a helper function like `cross_val_score`.

---

## 3. `sklearn.model_selection.cross_val_score`

### Purpose
Evaluates a model's performance using cross-validation in one line of code, without writing manual loops.

### Syntax
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator, X, y, cv=5, scoring='accuracy')
```

### Common Arguments
- `estimator`: The machine learning model you want to evaluate (e.g., `LogisticRegression()`).
- `X`: The feature data.
- `y`: The target labels.
- `cv`: Determines the cross-validation splitting strategy. If an integer is passed, it uses `KFold` or `StratifiedKFold`.
- `scoring` (string): The metric to evaluate the model (e.g., `'accuracy'`, `'neg_mean_squared_error'`, `'r2'`).

### Return Type
Returns an array of scores of the estimator for each run of the cross validation.

### Best Practices
Always check the average (`scores.mean()`) and the variance (`scores.std()`) of the output array to see how stable your model is across different data splits.
