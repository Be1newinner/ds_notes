# Non-Linear Regression

## Learning Objective
Students will learn how to model relationships that are not straight lines using Polynomial Regression and Support Vector Regression (SVR).

## What Is This Topic?
While simple linear regression fits a straight line, non-linear regression models can bend and curve to capture more complex patterns in the data (like exponential growth, cyclical patterns, or diminishing returns).

## Why This Topic Matters
In reality, very few relationships are perfectly linear. A company's revenue might grow exponentially at first and then plateau. A car's value drops rapidly in the first year and then depreciates slower. Non-linear models capture these real-world phenomena accurately.

## Core Intuition
- **Polynomial Regression**: Instead of just using $x$ to predict $y$, we create new features like $x^2$ and $x^3$. The model is still "linear" in the mathematical sense, but it draws a curve when plotted against the original $x$.
- **Support Vector Regression (SVR)**: Imagine a straight tube (a margin) around your data points. SVR tries to fit as many data points as possible *inside* this tube, ignoring errors smaller than a certain threshold. By using a "kernel trick," it can warp the space to draw complex non-linear tubes.

## Key Concepts
- **Degree (Polynomial)**: The highest power added. Degree 2 makes a U-shape (parabola). Degree 3 makes an S-shape.
- **Kernel Trick (SVR)**: A mathematical shortcut that transforms data into a higher dimension where a non-linear relationship looks linear.
- **Epsilon ($\epsilon$) Margin (SVR)**: The width of the tube where no penalty is given for errors.

## Important Parameters / Options / Settings
- `degree` (PolynomialFeatures): Controls flexibility. High degree = high chance of overfitting.
- `kernel` (SVR): Usually `'rbf'` (Radial Basis Function) for non-linear relationships.
- `C` (SVR): Regularization parameter. High `C` means strict fitting (risk of overfitting), low `C` means a smoother, more generalized curve.

## Real-World Uses
- Predicting the spread of a viral trend or disease (exponential phase, plateau phase).
- Modeling the relationship between vehicle speed and fuel efficiency (optimal at a certain speed, worse if slower or faster).

## Advantages
- **Polynomial**: Easy to implement using standard linear regression underneath.
- **SVR**: Highly robust to outliers (due to the epsilon margin).

## Limitations
- **Polynomial**: Extremely prone to overfitting if the degree is too high. Extrapolates very poorly outside the training data.
- **SVR**: Computationally expensive on large datasets. Requires careful feature scaling. Hard to interpret coefficients.

## Common Mistakes
- Using a high-degree polynomial (e.g., degree=10) that perfectly hits every training point but fails completely on new data.
- Forgetting to scale data before using SVR (it relies heavily on distance calculations).

## Code References
- `code/example-01-basic-polynomial.py`
- `code/example-02-intermediate-svr.py`


---

## Method Options: Non-Linear Regression

### `sklearn.preprocessing.PolynomialFeatures`

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

### `sklearn.svm.SVR`

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

---

## Practice: Non-Linear Regression

### Conceptual Questions
1. If you increase the `degree` of Polynomial Regression to 20, what will likely happen to your training error and your test error? What is this phenomenon called?
2. Why is the 'kernel trick' in SVR so important? 
3. How does the `epsilon` parameter in SVR act differently from the `alpha` penalty in Ridge/Lasso?

### Coding Tasks

#### Task 1: Tuning Polynomial Degrees
1. Load a non-linear dataset (e.g., a subset of a housing dataset focusing on Age vs Price).
2. Create a loop that fits a `PolynomialFeatures` + `LinearRegression` pipeline for degrees 1, 2, 3, 5, and 10.
3. For each degree, calculate both the Training MSE and Test MSE.
4. Plot the MSEs on a graph (X-axis: Degree, Y-axis: Error). At what degree does the model start overfitting?

#### Task 2: Understanding SVR Scaling
1. Use the `make_regression` function to generate data.
2. Train an `SVR(kernel='rbf')` model directly on the raw data and record the R2 score.
3. Apply `StandardScaler` to both X and y.
4. Train a new `SVR(kernel='rbf')` model on the scaled data.
5. Inverse transform your predictions and compare the R2 score with the unscaled model. Document the difference in performance.

---

## Interview Questions: Non-Linear Regression

### Beginner Level
1. **What is the main difference between Linear Regression and Polynomial Regression?**
   - *Expected Answer*: Linear regression fits a straight line. Polynomial regression introduces higher-degree terms (like $x^2$, $x^3$) allowing the model to fit a curve to the data, while still being solved using linear regression techniques.

2. **What does SVR stand for?**
   - *Expected Answer*: Support Vector Regression. It is the regression counterpart to Support Vector Machines (SVM) used for classification.

### Intermediate Level
3. **What is the risk of using a very high degree in Polynomial Regression?**
   - *Expected Answer*: Overfitting. The curve will pass through almost every single training data point, capturing all the noise. As a result, it will perform terribly on new, unseen data (high variance).

4. **Explain the $\epsilon$-tube (epsilon-tube) in SVR.**
   - *Expected Answer*: In SVR, the model tries to fit a tube with a radius of $\epsilon$ around the data points. Any errors that fall *inside* this tube are ignored (penalty is zero). The model only penalizes points that fall outside the tube.

5. **Why is scaling absolutely crucial for SVR but not necessarily for ordinary Linear Regression (without regularization)?**
   - *Expected Answer*: SVR uses a distance metric (especially with the RBF kernel) to calculate the similarity between data points. If features are on different scales, features with larger ranges will dominate the distance calculations, ruining the model's performance. OLS linear regression just calculates slopes and intercepts analytically, so scale doesn't affect the final line's predictive power.

### Advanced / Practical Level
6. **If your SVR model is underfitting, which parameter should you change?**
   - *Expected Answer*: You could increase `C` (which increases the penalty for points outside the margin, forcing the model to fit the training data tighter). You could also decrease `epsilon` to narrow the tube, making the model more sensitive to small errors. Finally, if using RBF, you might adjust `gamma`.

7. **How does the 'kernel trick' work in simple terms?**
   - *Expected Answer*: It mathematically transforms data points into a higher-dimensional space where a complex, non-linear relationship becomes a simple, linear relationship. It does this by calculating the "similarity" between points without actually performing the expensive calculations to map them to the higher dimension.

---

## Python Code Examples

### `example-01-basic-polynomial.py`

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 1. Generate Non-Linear Data (e.g., Company Growth over 10 years)
np.random.seed(42)
years = np.arange(1, 11).reshape(-1, 1) # X
# Revenue grows quadratically, with some noise
revenue = 10 + 2 * (years**2) + np.random.normal(0, 15, size=(10, 1)) # y

# 2. Try fitting a standard straight line first (to see it fail)
linear_model = LinearRegression()
linear_model.fit(years, revenue)
y_pred_linear = linear_model.predict(years)

# 3. Apply Polynomial Transformation
# Create a transformer that will add x^2 features
poly = PolynomialFeatures(degree=2, include_bias=False)
years_poly = poly.fit_transform(years)

# Print to show students what happened
print("Original X (Years):\n", years[:3])
print("\nTransformed X (Years, Years^2):\n", years_poly[:3])

# 4. Fit a Linear Regression model on the Transformed Data
poly_model = LinearRegression()
poly_model.fit(years_poly, revenue)
y_pred_poly = poly_model.predict(years_poly)

# 5. Visualize the difference
plt.figure(figsize=(10, 6))
plt.scatter(years, revenue, color='black', label='Actual Data')
plt.plot(years, y_pred_linear, color='red', linestyle='--', label='Standard Linear Fit (Underfit)')
plt.plot(years, y_pred_poly, color='blue', linewidth=2, label='Polynomial Fit (Degree 2)')

plt.title('Company Revenue Growth: Linear vs Polynomial Fit')
plt.xlabel('Years')
plt.ylabel('Revenue (Millions)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Demonstrate what happens if we extrapolate (predict year 15)
year_15 = np.array([[15]])
year_15_poly = poly.transform(year_15)

pred_linear = linear_model.predict(year_15)[0][0]
pred_poly = poly_model.predict(year_15_poly)[0][0]

print(f"\nPrediction for Year 15:")
print(f"Linear Model Predicts: ${pred_linear:.2f} Million")
print(f"Polynomial Model Predicts: ${pred_poly:.2f} Million")
```

### `example-02-intermediate-svr.py`

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# 1. Generate Non-Linear Data (Sine wave with noise)
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel()
# Add noise to targets
y[::5] += 1 * (0.5 - np.random.rand(16))

# 2. CRITICAL: SVR Requires Data Scaling!
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# 3. Train SVR Models with different Kernels
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_poly = SVR(kernel='poly', C=100, degree=3, epsilon=0.1, coef0=1)
svr_lin = SVR(kernel='linear', C=100)

svr_rbf.fit(X_scaled, y_scaled)
svr_poly.fit(X_scaled, y_scaled)
svr_lin.fit(X_scaled, y_scaled)

# 4. Predict
# We use the scaled X to predict, then inverse transform to get the real y values back
y_rbf = scaler_y.inverse_transform(svr_rbf.predict(X_scaled).reshape(-1, 1)).ravel()
y_poly = scaler_y.inverse_transform(svr_poly.predict(X_scaled).reshape(-1, 1)).ravel()
y_lin = scaler_y.inverse_transform(svr_lin.predict(X_scaled).reshape(-1, 1)).ravel()

# 5. Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='black', label='Data')
plt.plot(X, y_rbf, color='red', lw=2, label='RBF model (Non-linear)')
plt.plot(X, y_poly, color='blue', lw=2, label='Polynomial model (degree 3)')
plt.plot(X, y_lin, color='green', lw=2, label='Linear model')

plt.title('Support Vector Regression (SVR)')
plt.xlabel('X data')
plt.ylabel('y value')
plt.legend()
plt.show()

print("Notice how the RBF kernel bends to perfectly fit the sine wave shape, while the linear kernel fails completely.")
```

### `example-03-real-world-svr-tuning.py`

```python
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
import numpy as np

# 1. Load Data
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names).iloc[:1500]
y = california.target[:1500]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Crucial: Scale Data for SVR
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

# 3. Setup SVR and RandomizedSearchCV
svr = SVR(kernel='rbf')

param_dist = {
    'C': np.logspace(-2, 2, 5), # [0.01, 0.1, 1.0, 10.0, 100.0]
    'gamma': ['scale', 'auto', 0.1, 1.0],
    'epsilon': [0.01, 0.1, 0.5]
}

print("Running Randomized Search for SVR Tuning...")
random_search = RandomizedSearchCV(
    estimator=svr,
    param_distributions=param_dist,
    n_iter=10, # Try 10 random combinations
    cv=3,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_scaled, y_train_scaled)

print("\n--- Tuning Results ---")
print(f"Best Parameters: {random_search.best_params_}")

# 4. Evaluate the best model
best_svr = random_search.best_estimator_

# Predict on scaled test data
y_pred_scaled = best_svr.predict(X_test_scaled)

# Inverse transform to get real predictions
y_pred_real = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

rmse = mean_squared_error(y_test, y_pred_real, squared=False)
print(f"Final Test RMSE (unscaled): ${rmse * 100000:.2f}")
```
