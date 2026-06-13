# Linear Regression and Regularization

## Learning Objective
Students will understand the core mechanism of fitting a straight line to data, handling multiple variables, and applying regularization (Ridge, Lasso) to prevent overfitting.

## What Is This Topic?
Linear Regression predicts a continuous numerical output by drawing a "line of best fit" through the data points. Regularization is an advanced form of linear regression that adds a penalty to the model to keep it simple and prevent it from memorizing the noise in the data.

## Why This Topic Matters
It is the most fundamental algorithm in machine learning. It provides a baseline for predictive modeling, is highly interpretable, and is widely used in finance, economics, and business forecasting.

## Core Intuition
Imagine you have data points scattered on a graph showing house size vs. house price. Linear regression draws a straight line through the middle of these points so that the total distance from the points to the line is as small as possible. Regularization (like putting a speed limit on a car) prevents the model's coefficients (the slope of the line) from becoming too extreme.

## Key Concepts
- **Weights (Coefficients)**: How much each feature influences the prediction.
- **Bias (Intercept)**: The starting baseline prediction if all features were zero.
- **Cost Function (MSE)**: The metric the model tries to minimize (the error).
- **Gradient Descent**: The optimizer that slowly adjusts the line to find the best fit.
- **L1 Regularization (Lasso)**: Can shrink some weights exactly to zero, doing automatic feature selection.
- **L2 Regularization (Ridge)**: Shrinks weights to be small but rarely exactly zero.

## Step-by-Step Explanation
1. Initialize the line with random weights and a bias.
2. Make predictions for all data points.
3. Calculate the Mean Squared Error (MSE) between predictions and actual values.
4. Use an optimizer to adjust the weights and bias slightly to lower the error.
5. Repeat until the error stops decreasing (convergence).

## Important Parameters / Options / Settings
- `fit_intercept`: Should the model calculate the baseline (intercept)? Almost always `True`.
- `alpha` (in Ridge/Lasso): The strength of the regularization penalty. Higher `alpha` means a simpler model.

## Output / Result Interpretation
The output is a continuous numerical value (e.g., predicted price = $250,000). The coefficients tell you the relationship: a coefficient of 50 for 'size' means for every 1 sq ft increase, the price goes up by $50.

## Real-World Uses
- Predicting next quarter's revenue based on marketing spend across channels.
- Estimating the yield of a crop based on rainfall and fertilizer amount.
- Determining the price elasticity of a product.

## Advantages
- Fast to train and predict.
- Highly interpretable (you can explain exactly how a prediction was made).
- Works very well when the relationship is genuinely linear.

## Limitations
- Cannot capture complex, non-linear relationships without feature engineering.
- Very sensitive to outliers.
- Assumes features are independent (multicollinearity can cause issues).

## Common Mistakes
- Not scaling data before using Ridge or Lasso.
- Ignoring outliers that pull the line of best fit away from the majority of data.
- Applying linear regression when the relationship is clearly a curve.

## Related Methods
- Polynomial Regression (adds non-linear terms).
- Logistic Regression (used for classification, not regression).

## Code References
- `code/example-01-basic-simple-linear.py` — simple introduction example
- `code/example-02-intermediate-multiple-linear.py` — handling multiple features
- `code/example-03-real-world-regularization.py` — applying Ridge/Lasso to real data


---

## Method Options: Linear Models in Scikit-Learn

This file covers the key classes and methods used for Linear Regression and Regularization in Scikit-Learn.

### `sklearn.linear_model.LinearRegression`

**Purpose**: Performs standard Ordinary Least Squares (OLS) Linear Regression.

**Syntax**:
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
```

**Common Arguments**:
- `fit_intercept` (bool, default `True`): Whether to calculate the intercept. Set to `False` if data is already centered around the origin.
- `n_jobs` (int, default `None`): Number of jobs to use for computation. `-1` uses all processors (useful for huge datasets or multiple targets).

**Common Attributes**:
- `coef_`: Array of shape (n_features,). The estimated coefficients for the features.
- `intercept_`: Float. The independent term in the linear model.

**Typical Workflow**:
```python
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model.coef_, model.intercept_)
```

---

### `sklearn.linear_model.Ridge`

**Purpose**: Linear regression with L2 regularization. Useful for handling multicollinearity and preventing overfitting.

**Syntax**:
```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
```

**Common Arguments**:
- `alpha` (float, default `1.0`): Regularization strength. Larger values specify stronger regularization.
- `solver` (str, default `'auto'`): Algorithm to use in the computational routines.
- `random_state` (int): Seed for reproducibility when using stochastic solvers.

**Common Attributes**:
- `coef_`, `intercept_` (same as LinearRegression).

**Important Note**: Data **must** be scaled (e.g., using `StandardScaler`) before applying Ridge, as the penalty depends on the scale of the features.

---

### `sklearn.linear_model.Lasso`

**Purpose**: Linear regression with L1 regularization. Useful for feature selection because it can shrink coefficients exactly to zero.

**Syntax**:
```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.1)
```

**Common Arguments**:
- `alpha` (float, default `1.0`): Regularization strength. Higher values push more coefficients to zero.
- `max_iter` (int, default `1000`): Maximum number of iterations for the solver.

---

### Output / Return Type
The `predict(X)` method returns a 1D numpy array of shape (n_samples,) containing the continuous numerical predictions.

### Common Mistakes
- **Forgetting to Scale Data**: Applying Ridge or Lasso without standardizing features causes features with larger scales to be penalized unfairly.
- **Misinterpreting Coefficients without Scaling**: You cannot compare the importance of `coef_` values if the features have different units (e.g., Age in years vs. Salary in dollars) unless the data was scaled first.

---

## Practice: Linear Models

### Conceptual Questions
1. If you fit a Linear Regression model and the Mean Squared Error is very high on both training and test data, is the model suffering from high bias or high variance?
2. Why is it dangerous to use a standard Linear Regression model on a dataset that has severe outliers?
3. What is the difference between Lasso and Ridge regression in terms of how they handle the feature coefficients?
4. If you have 10,000 features but only 100 of them are actually useful for prediction, which regularization technique (Lasso or Ridge) would be better and why?

### Coding Tasks

#### Task 1: Simple Linear Regression
1. Generate synthetic data using `sklearn.datasets.make_regression` with `n_features=1` and `noise=20`.
2. Split the data into train and test sets.
3. Train a `LinearRegression` model.
4. Predict on the test set and calculate the Mean Absolute Error (MAE).
5. Plot the original data points (scatter plot) and the regression line (line plot) using Matplotlib.

#### Task 2: Impact of Scaling on Regularization
1. Load a dataset with differently scaled features (e.g., California Housing dataset).
2. Train a `Ridge(alpha=10)` model on the raw, unscaled training data and record the test R2 score.
3. Apply `StandardScaler` to the training data, scale the test data, and train a new `Ridge(alpha=10)` model.
4. Compare the R2 scores and the model coefficients before and after scaling. What do you observe?

#### Task 3: Feature Selection with Lasso
1. Create a dataset with 50 features using `make_regression` (set `n_informative=5`).
2. Scale the data.
3. Train a `Lasso` regression model with an appropriate `alpha` (e.g., 0.5).
4. Print out the model coefficients. How many coefficients were shrunk exactly to zero?

---

## Interview Questions: Linear Models

### Beginner Level
1. **What is Linear Regression?**
   - *Expected Answer*: An algorithm that models the linear relationship between independent variables and a continuous dependent variable by fitting a straight line (or hyperplane) that minimizes the sum of squared errors.

2. **What is the purpose of the intercept/bias term?**
   - *Expected Answer*: It represents the predicted value when all independent variables are exactly zero. Without it, the line would be forced to pass through the origin (0,0), which often leads to a poor fit.

3. **What does a coefficient in Multiple Linear Regression mean?**
   - *Expected Answer*: Assuming no multicollinearity, a coefficient represents the change in the target variable for a one-unit change in that specific feature, while holding all other features constant.

### Intermediate Level
4. **What are the key assumptions of Linear Regression?**
   - *Expected Answer*: 
     1. Linearity: The relationship is linear.
     2. Independence: Residuals are independent.
     3. Homoscedasticity: Constant variance of residuals.
     4. Normality: Residuals are normally distributed.
     5. No multicollinearity: Features are not highly correlated with each other.

5. **Explain L1 and L2 Regularization.**
   - *Expected Answer*: Both add a penalty to the cost function to prevent overfitting by keeping weights small. L1 (Lasso) uses the absolute value of the weights and can push weights to exactly zero (feature selection). L2 (Ridge) uses the squared value of the weights, penalizing large weights heavily but rarely shrinking them to zero.

6. **Why do we need to scale data before using Ridge or Lasso?**
   - *Expected Answer*: The regularization penalty is based on the magnitude of the coefficients. If features are on different scales (e.g., 1-10 vs 1000-100000), features with smaller numerical ranges will need larger coefficients to have an effect, and thus will be unfairly penalized by the regularization term.

### Advanced / Practical Level
7. **If two features in your dataset are perfectly correlated, what happens to your Linear Regression model?**
   - *Expected Answer*: This is perfect multicollinearity. The math breaks down (the matrix becomes non-invertible for OLS). The coefficients become highly unstable and uninterpretable. Ridge regression can help mitigate this by distributing the weights among the correlated features.

8. **You ran a Lasso regression and it dropped a feature that the business domain experts swear is highly predictive. What might have happened?**
   - *Expected Answer*: Lasso arbitrarily selects one feature from a group of highly correlated features and shrinks the others to zero. If the dropped feature was highly correlated with another feature that Lasso kept, it will appear "unimportant" to Lasso, even though it possesses predictive power.

9. **Output Interpretation**: You predict house prices. The coefficient for "number of bathrooms" is 15,000, and for "distance to city center (miles)" is -5,000. Interpret this.
   - *Expected Answer*: For every additional bathroom, the house price is predicted to increase by $15,000, assuming all other factors remain constant. For every extra mile away from the city center, the house price decreases by $5,000.

---

## Python Code Examples

### `example-01-basic-simple-linear.py`

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Create a simple dataset: Study hours vs. Exam Score
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Exam_Score': [45, 50, 60, 65, 70, 78, 85, 88, 92, 98]
}

df = pd.DataFrame(data)

# Features (X) must be a 2D array, Target (y) is 1D
X = df[['Hours_Studied']] 
y = df['Exam_Score']

# 2. Initialize and Train the Model
model = LinearRegression()
model.fit(X, y)

# 3. Make Predictions
y_pred = model.predict(X)

# 4. Evaluate the Model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Model Intercept (Bias): {model.intercept_:.2f}")
print(f"Model Coefficient (Weight): {model.coef_[0]:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Interpretation: 
# The coefficient tells us how many points the score increases per hour of study.
# The intercept is the predicted score if someone studies for 0 hours.

# 5. Visualize the Line of Best Fit
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line (Best Fit)')
plt.title('Study Hours vs. Exam Score')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Prediction for a new student who studied 7.5 hours
new_hours = pd.DataFrame({'Hours_Studied': [7.5]})
prediction = model.predict(new_hours)
print(f"\nPredicted score for 7.5 hours of study: {prediction[0]:.2f}")
```

### `example-02-intermediate-multiple-linear.py`

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Create a dataset with multiple features
# Predicting employee salary based on experience, performance score, and training hours
np.random.seed(42)
n_samples = 200

experience_years = np.random.uniform(1, 15, n_samples)
performance_score = np.random.uniform(1, 10, n_samples)
training_hours = np.random.uniform(10, 100, n_samples)

# Base salary + 5k per year of exp + 2k per performance point + 100 per training hour + noise
salary = 40000 + (5000 * experience_years) + (2000 * performance_score) + (100 * training_hours) + np.random.normal(0, 5000, n_samples)

df = pd.DataFrame({
    'Experience_Years': experience_years,
    'Performance_Score': performance_score,
    'Training_Hours': training_hours,
    'Salary': salary
})

print("Dataset Preview:")
print(df.head(), "\n")

# 2. Split data into Features (X) and Target (y)
X = df[['Experience_Years', 'Performance_Score', 'Training_Hours']]
y = df['Salary']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the Multiple Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Make Predictions
y_pred = model.predict(X_test)

# 5. Evaluate and Interpret
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"R-squared: {r2:.4f}\n")

print("--- Model Interpretation ---")
print(f"Base Salary (Intercept): ${model.intercept_:.2f}")

# Pair features with their learned coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient (Impact on Salary)': model.coef_
})

print("\nCoefficients:")
print(coefficients)
print("\nInterpretation: For every 1 unit increase in 'Experience_Years', salary is predicted to increase by ~$5000, holding other factors constant.")
```

### `example-03-real-world-regularization.py`

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate a dataset with some "useless" or highly correlated features (multicollinearity)
np.random.seed(42)
n_samples = 500

# True useful features
house_size = np.random.uniform(800, 4000, n_samples)
num_rooms = np.round(house_size / 400 + np.random.normal(0, 0.5, n_samples))
age = np.random.uniform(0, 50, n_samples)

# Redundant/Noisy features
size_in_cm = house_size * 929.03  # Perfectly correlated with house_size
random_noise_1 = np.random.normal(0, 100, n_samples)
random_noise_2 = np.random.normal(0, 50, n_samples)

# Target: Price
price = 50000 + (150 * house_size) - (1000 * age) + np.random.normal(0, 20000, n_samples)

df = pd.DataFrame({
    'House_Size_sqft': house_size,
    'Num_Rooms': num_rooms,
    'Age_Years': age,
    'Size_cm2': size_in_cm,
    'Noise_1': random_noise_1,
    'Noise_2': random_noise_2
})

X = df
y = price

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. CRITICAL STEP: Scale the data before Regularization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Models
# A. Standard OLS
ols = LinearRegression()
ols.fit(X_train_scaled, y_train)

# B. Ridge (L2 Penalty)
ridge = Ridge(alpha=100.0) # alpha is regularization strength
ridge.fit(X_train_scaled, y_train)

# C. Lasso (L1 Penalty) - Good for feature selection
lasso = Lasso(alpha=1000.0)
lasso.fit(X_train_scaled, y_train)

# 5. Evaluate and Compare
models = {'OLS': ols, 'Ridge': ridge, 'Lasso': lasso}

for name, model in models.items():
    pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, pred)
    print(f"\n--- {name} Model ---")
    print(f"Test R-squared: {r2:.4f}")
    
    # Print coefficients to see the effect of regularization
    print("Coefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"  {feature}: {coef:.2f}")

print("\n--- Summary of Regularization Effects ---")
print("1. OLS struggles because 'House_Size_sqft' and 'Size_cm2' are perfectly correlated. It gives wild, arbitrary weights to them.")
print("2. Ridge distributes the weight somewhat evenly between correlated features and shrinks noise.")
print("3. Lasso completely drops 'Size_cm2' and the noise columns (coefficients = 0.00), performing automatic feature selection!")
```
