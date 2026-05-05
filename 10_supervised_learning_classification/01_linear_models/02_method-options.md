# Method Options: Logistic Regression in Scikit-Learn

This document explains the primary tools used to implement Logistic Regression in Python.

## `sklearn.linear_model.LogisticRegression`

### Purpose
Implements logistic regression. It can handle both binary classification and multiclass classification (via One-vs-Rest or Multinomial setups).

### Syntax
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=100)
```

### Common Arguments
- **`penalty`** (`{'l1', 'l2', 'elasticnet', 'none'}`, default=`'l2'`): Determines the type of regularization to apply. Regularization prevents overfitting by penalizing large coefficients.
- **`C`** (`float`, default=`1.0`): Inverse of regularization strength. Smaller values specify stronger regularization. Must be a positive float.
- **`solver`** (`{'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}`, default=`'lbfgs'`): The algorithm to use for optimization.
  - `'liblinear'`: Good for small datasets.
  - `'lbfgs'`: Good default.
  - `'saga'`: Good for very large datasets and supports `'l1'` penalty.
- **`max_iter`** (`int`, default=`100`): Maximum number of iterations taken for the solvers to converge. If your model throws a convergence warning, increase this number (e.g., `max_iter=1000`).
- **`class_weight`** (`dict` or `'balanced'`, default=`None`): Crucial for imbalanced datasets. `'balanced'` automatically adjusts weights inversely proportional to class frequencies.
- **`multi_class`** (`{'auto', 'ovr', 'multinomial'}`, default=`'auto'`): Strategy for multiclass. `'ovr'` means One-vs-Rest.

### Common Attributes / Properties
- **`coef_`**: An array of shape `(n_classes, n_features)`. The learned weights for the features.
- **`intercept_`**: An array of shape `(n_classes,)`. The bias term added to the decision function.
- **`classes_`**: The list of class labels known to the classifier.

### Common Methods
- **`fit(X, y)`**: Trains the model on the data.
- **`predict(X)`**: Returns the predicted class labels (e.g., `[0, 1, 1, 0]`).
- **`predict_proba(X)`**: Returns the probability estimates for all classes. Shape is `(n_samples, n_classes)`. For binary, column 0 is prob of class 0, column 1 is prob of class 1.
- **`score(X, y)`**: Returns the mean accuracy on the given test data and labels.

### Typical Workflow
1. Prepare and scale features (`StandardScaler`).
2. Initialize `LogisticRegression(C=1.0, max_iter=1000)`.
3. Call `.fit(X_train, y_train)`.
4. Check probabilities with `.predict_proba(X_test)`.
5. Get hard predictions with `.predict(X_test)`.
6. Evaluate with accuracy, confusion matrix, or classification report.

### Common Mistakes
- **Forgetting to scale data**: Logistic Regression by default uses `l2` penalty. Regularization assumes all features are on the same scale. If they aren't, the penalty affects features unevenly.
- **Ignoring the `ConvergenceWarning`**: If the solver doesn't converge, the coefficients are garbage. Always increase `max_iter` or scale your data if you see this warning.
