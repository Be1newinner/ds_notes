# Method Options: Support Vector Machines in Scikit-Learn

This document explains the tools used to implement SVMs in Python.

## `sklearn.svm.SVC` (Support Vector Classification)

### Purpose
The primary implementation of SVM for classification. It supports both linear and non-linear kernels.

### Syntax
```python
from sklearn.svm import SVC
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
```

### Common Arguments
- **`C`** (`float`, default=`1.0`): Regularization parameter. The strength of the regularization is inversely proportional to `C`. Must be strictly positive. The penalty is a squared l2 penalty.
- **`kernel`** (`{'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}`, default=`'rbf'`): Specifies the kernel type to be used in the algorithm.
  - `'linear'`: Standard straight-line decision boundary.
  - `'poly'`: Polynomial boundary.
  - `'rbf'`: Radial Basis Function. Good for complex, non-linear boundaries. (Default)
- **`gamma`** (`{'scale', 'auto'}`, or `float`, default=`'scale'`): Kernel coefficient for `'rbf'`, `'poly'` and `'sigmoid'`. 
  - If `gamma='scale'` (default), it uses `1 / (n_features * X.var())`.
  - Intuitively: higher gamma = more complex boundary (risk of overfitting).
- **`probability`** (`bool`, default=`False`): Whether to enable probability estimates. **Note:** This must be enabled prior to calling `fit`, and it slows down the training considerably because it uses 5-fold cross-validation internally.

### Common Attributes
- **`support_vectors_`**: Array of the support vectors.
- **`n_support_`**: Number of support vectors for each class.
- **`coef_`**: Weights assigned to the features (only available when `kernel="linear"`).

### Typical Workflow
1. **Scale the data:** SVMs are highly sensitive to feature scales. Use `StandardScaler`.
2. Initialize `SVC`.
3. Fit the model.
4. Predict.

---

## `sklearn.svm.LinearSVC`

### Purpose
Similar to `SVC` with parameter `kernel='linear'`, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and **scales much better to large numbers of samples**.

### Syntax
```python
from sklearn.svm import LinearSVC
model = LinearSVC(C=1.0, dual=False, max_iter=1000)
```

### Common Arguments
- **`C`** (`float`, default=`1.0`): Regularization parameter.
- **`dual`** (`bool`, default=`True`): Select the algorithm to either solve the dual or primal optimization problem. Prefer `dual=False` when `n_samples > n_features`.
- **`max_iter`** (`int`, default=`1000`): The maximum number of iterations to be run.

### When to use `LinearSVC` vs `SVC(kernel='linear')`?
- If your dataset has more than ~10,000 rows and you want a linear boundary, use `LinearSVC`. It is dramatically faster.
- If you need non-linear boundaries (`rbf`, `poly`), you *must* use `SVC`.
