# Linear Models for Classification: Logistic Regression

## Learning Objective
Understand how to adapt linear models for classification tasks using Logistic Regression. Learn the underlying mathematics intuitively and know when to apply it.

## What Is This Topic?
Logistic Regression is a foundational classification algorithm. Despite its name containing "regression," it is used for predicting categorical outcomes (like Yes/No, True/False). It calculates the probability that a given data point belongs to a specific class.

## Why This Topic Matters
Logistic regression is the building block for many advanced algorithms (like Neural Networks). It is highly interpretable, meaning you can explain exactly *why* a decision was made. It's often the first model tried in industry due to its simplicity and speed.

## Core Intuition
Instead of drawing a straight line through the data (like in linear regression), logistic regression fits an S-shaped curve (the Sigmoid curve) to the data. This curve takes any real number and squashes it into a value between 0 and 1, representing a probability.

## Key Concepts
- **Sigmoid Function**: The mathematical function $S(x) = \frac{1}{1 + e^{-x}}$ that squashes output between 0 and 1.
- **Log-Odds**: Logistic regression models the logarithm of the odds (probability of event occurring divided by probability of it not occurring) as a linear combination of features.
- **Decision Boundary**: A threshold (usually 0.5) used to convert the predicted probability into a hard class label (0 or 1).

## Step-by-Step Explanation
1. **Linear Combination**: The model computes a weighted sum of the input features plus a bias (just like linear regression).
2. **Sigmoid Activation**: The weighted sum is passed through the Sigmoid function.
3. **Probability Output**: The result is a number between 0.0 and 1.0 (e.g., 0.85 means 85% chance of being class 1).
4. **Classification**: If the probability is $\ge 0.5$, predict Class 1. Otherwise, predict Class 0.

## Important Parameters / Options / Settings
- **`C` (Inverse of regularization strength)**: Smaller values specify stronger regularization to prevent overfitting.
- **`penalty`**: Used to specify the norm used in the penalization (`l1`, `l2`, `elasticnet`, `none`).
- **`solver`**: Algorithm to use in the optimization problem (`liblinear`, `lbfgs`, `saga`). SAGA is good for large datasets.

## Output / Result Interpretation
- **Coefficients (`coef_`)**: A positive coefficient means the feature increases the probability of Class 1. A negative coefficient decreases the probability.
- **Probabilities (`predict_proba`)**: Tells you *how confident* the model is, not just the final prediction.

## Real-World Uses
- **Medical Diagnosis**: Predicting if a tumor is malignant or benign based on size and characteristics.
- **Credit Scoring**: Predicting if a customer will default on a loan.
- **Marketing**: Predicting if a user will click on an ad (Click-Through Rate).

## Advantages
- Very fast to train and predict.
- Highly interpretable.
- Outputs calibrated probabilities, not just hard labels.
- Less prone to overfitting on small datasets.

## Limitations
- Assumes a linear relationship between features and the log-odds.
- Cannot solve non-linear problems easily without feature engineering (like polynomial features).
- Outperformed by tree-based models on complex tabular data.

## Common Mistakes
- Not scaling features. Regularization in logistic regression is sensitive to feature scales.
- Treating it as a regression problem just because of the name.
- Ignoring multicollinearity among features, which messes up the interpretability of coefficients.

## Related Methods
- Linear Support Vector Classification (LinearSVC).
- Naive Bayes.
- Neural Networks (a single neuron with a sigmoid activation *is* logistic regression).

## Code References
- `code/example-01-basic-logistic.py`
- `code/example-02-multiclass.py`
- `code/example-03-real-world-churn.py`


---

## Method Options: Logistic Regression in Scikit-Learn

This document explains the primary tools used to implement Logistic Regression in Python.

### `sklearn.linear_model.LogisticRegression`

#### Purpose
Implements logistic regression. It can handle both binary classification and multiclass classification (via One-vs-Rest or Multinomial setups).

#### Syntax
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=100)
```

#### Common Arguments
- **`penalty`** (`{'l1', 'l2', 'elasticnet', 'none'}`, default=`'l2'`): Determines the type of regularization to apply. Regularization prevents overfitting by penalizing large coefficients.
- **`C`** (`float`, default=`1.0`): Inverse of regularization strength. Smaller values specify stronger regularization. Must be a positive float.
- **`solver`** (`{'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}`, default=`'lbfgs'`): The algorithm to use for optimization.
  - `'liblinear'`: Good for small datasets.
  - `'lbfgs'`: Good default.
  - `'saga'`: Good for very large datasets and supports `'l1'` penalty.
- **`max_iter`** (`int`, default=`100`): Maximum number of iterations taken for the solvers to converge. If your model throws a convergence warning, increase this number (e.g., `max_iter=1000`).
- **`class_weight`** (`dict` or `'balanced'`, default=`None`): Crucial for imbalanced datasets. `'balanced'` automatically adjusts weights inversely proportional to class frequencies.
- **`multi_class`** (`{'auto', 'ovr', 'multinomial'}`, default=`'auto'`): Strategy for multiclass. `'ovr'` means One-vs-Rest.

#### Common Attributes / Properties
- **`coef_`**: An array of shape `(n_classes, n_features)`. The learned weights for the features.
- **`intercept_`**: An array of shape `(n_classes,)`. The bias term added to the decision function.
- **`classes_`**: The list of class labels known to the classifier.

#### Common Methods
- **`fit(X, y)`**: Trains the model on the data.
- **`predict(X)`**: Returns the predicted class labels (e.g., `[0, 1, 1, 0]`).
- **`predict_proba(X)`**: Returns the probability estimates for all classes. Shape is `(n_samples, n_classes)`. For binary, column 0 is prob of class 0, column 1 is prob of class 1.
- **`score(X, y)`**: Returns the mean accuracy on the given test data and labels.

#### Typical Workflow
1. Prepare and scale features (`StandardScaler`).
2. Initialize `LogisticRegression(C=1.0, max_iter=1000)`.
3. Call `.fit(X_train, y_train)`.
4. Check probabilities with `.predict_proba(X_test)`.
5. Get hard predictions with `.predict(X_test)`.
6. Evaluate with accuracy, confusion matrix, or classification report.

#### Common Mistakes
- **Forgetting to scale data**: Logistic Regression by default uses `l2` penalty. Regularization assumes all features are on the same scale. If they aren't, the penalty affects features unevenly.
- **Ignoring the `ConvergenceWarning`**: If the solver doesn't converge, the coefficients are garbage. Always increase `max_iter` or scale your data if you see this warning.

---

## Examples: Logistic Regression

Here is a breakdown of the Python examples provided in the `code/` directory.

### 1. Basic Binary Classification (`example-01-basic-logistic.py`)
- **Goal:** Introduce the fundamental usage of Logistic Regression.
- **Dataset:** A synthetic dataset representing a simple pass/fail scenario based on study hours and sleep.
- **Key Concepts Shown:** Model initialization, `fit()`, `predict()`, and `predict_proba()`.
- **Takeaway:** Shows how Logistic Regression outputs a probability and converts it into a hard 0 or 1 class prediction based on a 0.5 threshold.

### 2. Multiclass Classification (`example-02-multiclass.py`)
- **Goal:** Show how Logistic Regression handles more than two categories.
- **Dataset:** The classic Iris dataset (predicting 3 different flower species).
- **Key Concepts Shown:** `multi_class='multinomial'`, interpreting the shape of `predict_proba()` (which now outputs 3 probabilities that sum to 1), and `coef_` shapes.
- **Takeaway:** Logistic Regression is not limited to binary (Yes/No) questions.

### 3. Real-World Scenario: Customer Churn (`example-03-real-world-churn.py`)
- **Goal:** Apply Logistic Regression to a realistic business problem.
- **Dataset:** A simulated telecom customer dataset containing demographics, account info, and whether they canceled their subscription (Churn).
- **Key Concepts Shown:** 
  - Data scaling (StandardScaler) which is critical for regularized logistic regression.
  - Interpreting coefficients to drive business insights (e.g., "High monthly charges strongly increase the chance of churn").
  - Evaluating using a confusion matrix and classification report instead of just simple accuracy.
- **Takeaway:** How to translate mathematical model outputs into actionable business advice.

---

## Practice Exercises: Logistic Regression

These exercises are designed to test your conceptual understanding and coding skills.

### Conceptual Questions
1. Why is the sigmoid function necessary in Logistic Regression? What would happen if we just used a straight line (Linear Regression) to predict classes?
2. You fit a logistic regression model predicting if an email is spam (1) or not (0). The coefficient for the word "lottery" is `+2.5`. What does this mean in plain English?
3. If your model outputs a probability of `0.49`, what class will `.predict()` output by default? How might you change the threshold if you want to be extremely careful about missing Spam emails?

### Coding Tasks

#### Task 1: Basic Fit & Predict
Load the `breast_cancer` dataset from `sklearn.datasets`. Split it into 80% train and 20% test. Fit a Logistic Regression model and print the accuracy on the test set.

#### Task 2: The Importance of Scaling
Using the same breast cancer dataset:
1. Fit a Logistic Regression model *without* scaling the data. Print the accuracy.
2. Scale the data using `StandardScaler`.
3. Fit a new Logistic Regression model on the scaled data. Print the accuracy.
4. Compare the two accuracies and explain the difference.

#### Task 3: Changing the Decision Threshold
By default, `.predict()` uses a 0.5 threshold. 
1. Train a model on any binary dataset.
2. Use `.predict_proba()` to get the raw probabilities.
3. Write a small Python script to manually create predictions using a threshold of `0.8` (i.e., only predict class 1 if you are 80% sure). 
4. How does this change your false positives and false negatives?

---

## Interview Questions: Logistic Regression

These questions test your practical and theoretical knowledge of Logistic Regression, commonly asked in data science interviews.

### Beginner Questions
1. **What is the difference between Linear Regression and Logistic Regression?**
   *Hint:* Discuss the type of output (continuous vs. discrete probability), the line vs. the S-curve, and the use of the sigmoid function.
2. **What does the output of a Logistic Regression model represent?**
   *Hint:* It represents a probability (from 0 to 1) that the observation belongs to the positive class.
3. **What is the default threshold used in Logistic Regression to make a class prediction?**
   *Hint:* 0.5.

### Conceptual Questions
4. **Why is it called "regression" if it is used for classification?**
   *Hint:* Because under the hood, it fits a linear regression model to the *log-odds* of the probability.
5. **How does the sigmoid function work?**
   *Hint:* It takes any real-valued number and maps it to a value between 0 and 1 using the formula $1 / (1 + e^{-x})$.
6. **Can Logistic Regression handle non-linear decision boundaries?**
   *Hint:* Not natively. It is a linear classifier. However, you can engineer non-linear features (like $x^2$ or $x_1 * x_2$) to allow it to fit non-linear data.

### Practical Questions
7. **Why is feature scaling (like Standardization) important before training a Logistic Regression model in scikit-learn?**
   *Hint:* Scikit-learn applies L2 regularization by default. Regularization penalizes large coefficients. If features are on different scales, the penalty will disproportionately affect features with smaller absolute values.
8. **If you have a highly imbalanced dataset (e.g., 99% class 0, 1% class 1), what parameter in `LogisticRegression` should you change?**
   *Hint:* Set `class_weight='balanced'`.
9. **You train a Logistic Regression model and the solver throws a `ConvergenceWarning`. What does this mean and how do you fix it?**
   *Hint:* It means the optimization algorithm didn't find the minimum of the cost function in the allotted steps. Fix it by increasing `max_iter` or scaling the data.

### Output Interpretation
10. **If the coefficient for the feature "Age" is -0.5, what does that mean?**
    *Hint:* It means that as "Age" increases, the log-odds of the positive class decrease. In plain English, older people are less likely to belong to the positive class.

---

## Python Code Examples

### `example-01-basic-logistic.py`

```python
"""
Example 01: Basic Logistic Regression
Goal: Understand how to fit a basic Logistic Regression model for binary classification.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Create a simple synthetic dataset
# Scenario: Predicting if a student passes a test based on Hours Studied
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Passed': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}
df = pd.DataFrame(data)

# 2. Separate features (X) and target (y)
X = df[['Hours_Studied']]
y = df['Passed']

# 3. Initialize the model
# We use default settings (L2 regularization, C=1.0)
model = LogisticRegression()

# 4. Train the model
model.fit(X, y)

# 5. Make predictions on some new student data
new_students = pd.DataFrame({'Hours_Studied': [2.5, 4.5, 6.5]})
predictions = model.predict(new_students)

# predict_proba gives the actual probabilities [prob_Fail, prob_Pass]
probabilities = model.predict_proba(new_students)

print("--- Prediction Results ---")
for i, hours in enumerate(new_students['Hours_Studied']):
    prob_pass = probabilities[i][1] * 100
    pred_class = "Pass" if predictions[i] == 1 else "Fail"
    print(f"Hours Studied: {hours} | Prob of Passing: {prob_pass:.2f}% | Final Prediction: {pred_class}")

# 6. Look at the learned coefficients
print("\n--- Model Internals ---")
print(f"Coefficient (Weight for Hours_Studied): {model.coef_[0][0]:.4f}")
print(f"Intercept (Bias): {model.intercept_[0]:.4f}")
print("Notice the coefficient is positive: More hours studied increases the probability of passing.")
```

### `example-02-multiclass.py`

```python
"""
Example 02: Multiclass Logistic Regression
Goal: Use Logistic Regression to classify data into more than two categories.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the Iris dataset (3 classes of flowers)
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

print(f"Classes to predict: {target_names}")

# 2. Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize the model
# multi_class='multinomial' tells it to use a true multiclass loss function
# max_iter=200 is used because the default (100) might throw a ConvergenceWarning
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)

# 4. Train
model.fit(X_train, y_train)

# 5. Predict on test set
y_pred = model.predict(X_test)

# 6. Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on test set: {acc * 100:.2f}%\n")

print("--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=target_names))

# Let's look at predict_proba for the first test sample
probs = model.predict_proba(X_test[[0]])
print("\n--- Probability Output for the First Test Sample ---")
print(f"Prob Setosa: {probs[0][0]:.4f}")
print(f"Prob Versicolor: {probs[0][1]:.4f}")
print(f"Prob Virginica: {probs[0][2]:.4f}")
print("Notice that the probabilities sum to 1.0")
```

### `example-03-real-world-churn.py`

```python
"""
Example 03: Real-World Scenario - Customer Churn Prediction
Goal: Apply Logistic Regression to a business problem, including data scaling and coefficient interpretation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# 1. Simulate a Telecom Churn Dataset
np.random.seed(42)
n_samples = 1000

# Features: Tenure (months), Monthly Charge ($), Calls to Customer Service
tenure = np.random.randint(1, 72, n_samples)
monthly_charge = np.random.uniform(20, 120, n_samples)
customer_service_calls = np.random.randint(0, 6, n_samples)

# Create a "churn probability" driven by the features
# Lower tenure, higher charge, more service calls = higher chance of churn
log_odds = -0.05 * tenure + 0.02 * monthly_charge + 0.8 * customer_service_calls - 2
prob_churn = 1 / (1 + np.exp(-log_odds))
churn = (np.random.rand(n_samples) < prob_churn).astype(int)

df = pd.DataFrame({
    'Tenure_Months': tenure,
    'Monthly_Charge': monthly_charge,
    'Cust_Service_Calls': customer_service_calls,
    'Churn': churn
})

X = df.drop('Churn', axis=1)
y = df['Churn']

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. CRITICAL STEP: Scale the features
# Logistic regression uses L2 regularization by default, so scaling is mandatory!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train the model
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_scaled)
print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# 6. Business Interpretation of Coefficients
print("\n--- Feature Importance (Coefficients) ---")
# Because data is scaled, the magnitude of the coefficient shows importance
for feature, coef in zip(X.columns, model.coef_[0]):
    direction = "Increases" if coef > 0 else "Decreases"
    print(f"{feature:20} : {coef:>7.4f} ({direction} chance of Churn)")

print("\nInsight for Business Team:")
print("Customers with many customer service calls have the highest risk of churning.")
print("Customers who have been with us longer (higher tenure) have a lower risk of churning.")
```
