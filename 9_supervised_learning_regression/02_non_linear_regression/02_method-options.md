# Method Options: Non-Linear Regression

## `sklearn.preprocessing.PolynomialFeatures`

**Purpose**: Generates a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree.

**Syntax**:
```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
```

**Common Arguments**:
- `degree` (int, default `2`): The degree of the polynomial features.
- `include_bias` (bool, default `True`): If True, includes a bias column (all 1s). Usually set to `False` if passing the output directly to a `LinearRegression` model that calculates its own intercept.

**Typical Workflow**:
```python
# 1. Transform features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 2. Fit standard linear model on transformed features
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_poly, y_train)
```

---

## `sklearn.svm.SVR`

**Purpose**: Epsilon-Support Vector Regression.

**Syntax**:
```python
from sklearn.svm import SVR
model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
```

**Common Arguments**:
- `kernel` (str, default `'rbf'`): Specifies the kernel type. Options: `'linear'`, `'poly'`, `'rbf'`, `'sigmoid'`. `'rbf'` is the most common for non-linear data.
- `C` (float, default `1.0`): Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
- `epsilon` (float, default `0.1`): Specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.

**Important Note**: You **must** scale both `X` and `y` when using SVR. Unlike Random Forests or ordinary Linear Regression, distance-based algorithms are highly sensitive to unscaled data.

**Workflow**:
```python
from sklearn.preprocessing import StandardScaler

# Scale X and y
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
# y usually needs to be 2D array for scaling: y.values.reshape(-1, 1)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

model = SVR(kernel='rbf')
model.fit(X_scaled, y_scaled)
```
