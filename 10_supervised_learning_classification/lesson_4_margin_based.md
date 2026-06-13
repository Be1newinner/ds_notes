# Margin-Based Methods: Support Vector Machines (SVM)

## Learning Objective
Understand how Support Vector Machines classify data by finding the optimal hyperplane that maximizes the margin between classes, and learn how the "Kernel Trick" allows SVMs to solve complex non-linear problems.

## What Is This Topic?
A Support Vector Machine (SVM) is a powerful classification algorithm. Unlike Logistic Regression, which outputs probabilities based on a curve, SVM draws a strict line (or plane) through the data to separate the classes. It specifically looks for the line that leaves the maximum possible "gap" (margin) between the classes.

## Why This Topic Matters
Before Deep Learning became dominant, SVMs with non-linear kernels were considered the state-of-the-art algorithm for many classification tasks. They are highly effective in high-dimensional spaces (e.g., text categorization, image recognition) and are still widely used today for complex tabular data.

## Core Intuition
Imagine a table with red apples and green apples mixed together. You need to put a straight stick on the table to separate them. You could put the stick in many places, but the *best* place is right in the middle, as far away from both the red apples and the green apples as possible. That stick is the **hyperplane**, and the distance from the stick to the closest apples is the **margin**. The closest apples that "support" this margin are the **Support Vectors**.

## Key Concepts
- **Hyperplane**: The decision boundary. In 2D, it's a line. In 3D, it's a flat plane. In $N$-dimensions, it's an $N-1$ dimensional hyperplane.
- **Margin**: The distance between the hyperplane and the closest data points from either class.
- **Support Vectors**: The data points that lie exactly on the edge of the margin. *These are the only points the algorithm cares about.* If you remove all other data points, the hyperplane wouldn't change.
- **Hard Margin vs. Soft Margin**: A hard margin allows zero misclassifications (requires perfectly separable data). A soft margin allows some points to cross the line to prevent overfitting.
- **The Kernel Trick**: If data is not linearly separable in 2D, a Kernel function mathematically projects the data into 3D (or higher) where it *is* linearly separable, without actually doing the heavy computation.

## Important Parameters
- **`C` (Regularization parameter)**: Controls the trade-off between a wide margin and misclassification.
  - **High `C`**: Strict. "I don't want any misclassifications!" (Smaller margin, risks overfitting).
  - **Low `C`**: Relaxed. "I want a wide margin, even if I get a few points wrong." (Larger margin, risks underfitting).
- **`kernel`**: The function used to transform the data (`linear`, `poly`, `rbf`). `rbf` (Radial Basis Function) is the default and most popular for non-linear data.
- **`gamma`**: (Used in RBF kernel). Controls the "influence" of a single training point. High gamma means points only influence things very close to them (bumpy decision boundary). Low gamma means points have a far-reaching influence (smooth decision boundary).

## Advantages
- Extremely effective in high dimensional spaces.
- Memory efficient (only uses a subset of training points—the support vectors).
- Versatile, due to the ability to use different Kernel functions.

## Limitations
- **Does not scale well to large datasets** (training time is $O(n^2)$ to $O(n^3)$). Do not use on datasets with $>100,000$ rows.
- Highly sensitive to unscaled data.
- It does not directly provide probability estimates (though they can be calculated using expensive cross-validation via `probability=True`).
- Hard to interpret compared to Decision Trees or Logistic Regression.

## Related Methods
- Support Vector Regression (SVR) for continuous targets.
- Logistic Regression (similar to a linear SVM).

## Code References
- `code/example-01-basic-svm-linear.py`
- `code/example-02-svm-kernels.py`
- `code/example-03-real-world-medical.py`


---

## Method Options: Support Vector Machines in Scikit-Learn

This document explains the tools used to implement SVMs in Python.

### `sklearn.svm.SVC` (Support Vector Classification)

#### Purpose
The primary implementation of SVM for classification. It supports both linear and non-linear kernels.

#### Syntax
```python
from sklearn.svm import SVC
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
```

#### Common Arguments
- **`C`** (`float`, default=`1.0`): Regularization parameter. The strength of the regularization is inversely proportional to `C`. Must be strictly positive. The penalty is a squared l2 penalty.
- **`kernel`** (`{'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}`, default=`'rbf'`): Specifies the kernel type to be used in the algorithm.
  - `'linear'`: Standard straight-line decision boundary.
  - `'poly'`: Polynomial boundary.
  - `'rbf'`: Radial Basis Function. Good for complex, non-linear boundaries. (Default)
- **`gamma`** (`{'scale', 'auto'}`, or `float`, default=`'scale'`): Kernel coefficient for `'rbf'`, `'poly'` and `'sigmoid'`. 
  - If `gamma='scale'` (default), it uses `1 / (n_features * X.var())`.
  - Intuitively: higher gamma = more complex boundary (risk of overfitting).
- **`probability`** (`bool`, default=`False`): Whether to enable probability estimates. **Note:** This must be enabled prior to calling `fit`, and it slows down the training considerably because it uses 5-fold cross-validation internally.

#### Common Attributes
- **`support_vectors_`**: Array of the support vectors.
- **`n_support_`**: Number of support vectors for each class.
- **`coef_`**: Weights assigned to the features (only available when `kernel="linear"`).

#### Typical Workflow
1. **Scale the data:** SVMs are highly sensitive to feature scales. Use `StandardScaler`.
2. Initialize `SVC`.
3. Fit the model.
4. Predict.

---

### `sklearn.svm.LinearSVC`

#### Purpose
Similar to `SVC` with parameter `kernel='linear'`, but implemented in terms of liblinear rather than libsvm, so it has more flexibility in the choice of penalties and loss functions and **scales much better to large numbers of samples**.

#### Syntax
```python
from sklearn.svm import LinearSVC
model = LinearSVC(C=1.0, dual=False, max_iter=1000)
```

#### Common Arguments
- **`C`** (`float`, default=`1.0`): Regularization parameter.
- **`dual`** (`bool`, default=`True`): Select the algorithm to either solve the dual or primal optimization problem. Prefer `dual=False` when `n_samples > n_features`.
- **`max_iter`** (`int`, default=`1000`): The maximum number of iterations to be run.

#### When to use `LinearSVC` vs `SVC(kernel='linear')`?
- If your dataset has more than ~10,000 rows and you want a linear boundary, use `LinearSVC`. It is dramatically faster.
- If you need non-linear boundaries (`rbf`, `poly`), you *must* use `SVC`.

---

## Examples: Support Vector Machines

Here is a breakdown of the Python examples provided in the `code/` directory.

### 1. Basic Linear SVM (`example-01-basic-svm-linear.py`)
- **Goal:** Introduce the mechanics of a hard-margin vs. soft-margin SVM on linearly separable data.
- **Dataset:** A synthetic 2D dataset that can be cleanly separated with a straight line.
- **Key Concepts Shown:** 
  - Using `SVC(kernel='linear')`.
  - Extracting and visualizing the `support_vectors_`.
  - Demonstrating how changing the `C` parameter alters the margin size.
- **Takeaway:** The concept of the margin and how Support Vectors are the only points that dictate the boundary.

### 2. SVM Kernels (`example-02-svm-kernels.py`)
- **Goal:** Show why the Kernel Trick is the defining feature of SVMs.
- **Dataset:** The `make_circles` dataset—data points arranged in two concentric circles (impossible to separate with a straight line).
- **Key Concepts Shown:** 
  - Training a `linear` kernel (which completely fails).
  - Training an `rbf` (Radial Basis Function) kernel (which perfectly separates the circles).
- **Takeaway:** Kernels allow SVMs to draw complex, non-linear boundaries by mathematically projecting the data into higher dimensions.

### 3. Real-World Scenario: Medical Diagnosis (`example-03-real-world-medical.py`)
- **Goal:** Apply SVM to a complex dataset with many features.
- **Dataset:** The classic Breast Cancer Wisconsin dataset (predicting Malignant vs. Benign).
- **Key Concepts Shown:** 
  - The absolute necessity of `StandardScaler` for SVMs.
  - Setting `probability=True` to get probability scores (which SVMs don't do by default).
  - Interpreting the classification report.
- **Takeaway:** SVMs are incredibly powerful for high-dimensional medical data, provided the data is scaled properly.

---

## Practice Exercises: Support Vector Machines

These exercises are designed to test your conceptual understanding and coding skills.

### Conceptual Questions
1. Why does an SVM only care about the Support Vectors? If you deleted 90% of the training data that are far away from the margin, would the model change?
2. You train an SVM with an RBF kernel and achieve 100% training accuracy but 60% test accuracy. The model is overfitting. Which two hyperparameters (`C` and `gamma`) should you decrease/increase to simplify the model?
3. What is the main disadvantage of setting `probability=True` in `SVC`?

### Coding Tasks

#### Task 1: Linear vs RBF
1. Use `sklearn.datasets.make_moons(n_samples=200, noise=0.15)` to generate a non-linear dataset.
2. Scale the data.
3. Train two models: `SVC(kernel='linear')` and `SVC(kernel='rbf')`.
4. Compare their accuracy.

#### Task 2: The Effect of Gamma
Using the `make_moons` dataset from Task 1:
1. Train three different `SVC(kernel='rbf')` models with `gamma` set to `0.1`, `1.0`, and `10.0`. Keep `C=1.0`.
2. Look at the training accuracy vs testing accuracy for each. 
3. Based on the accuracy scores, what happens as `gamma` gets larger?

#### Task 3: LinearSVC for Speed
1. Generate a large, simple dataset using `sklearn.datasets.make_classification(n_samples=50000, n_features=20)`.
2. Time how long it takes to train `SVC(kernel='linear')` using the `time` module.
3. Time how long it takes to train `LinearSVC()`.
4. Compare the training times and their accuracies.

---

## Interview Questions: Support Vector Machines

### Beginner Questions
1. **Explain the goal of a Support Vector Machine in your own words.**
   *Hint:* The goal is to find a line (hyperplane) that separates different classes while maximizing the margin (the gap) between the line and the closest data points from each class.
2. **What are Support Vectors?**
   *Hint:* They are the data points closest to the hyperplane. They are the only points that dictate where the boundary is drawn.
3. **What is the "Kernel Trick"?**
   *Hint:* It's a mathematical technique that allows SVMs to classify non-linearly separable data by mapping it into a higher-dimensional space where it *is* linearly separable.

### Conceptual Questions
4. **Explain the role of the `C` parameter.**
   *Hint:* `C` is the regularization parameter. A high `C` creates a "hard margin" (penalizes misclassifications heavily, leading to a smaller margin and potential overfitting). A low `C` creates a "soft margin" (allows some misclassifications for the sake of a wider, more generalizable margin).
5. **How does the RBF kernel work intuitively?**
   *Hint:* Radial Basis Function measures the similarity between two points based on how close they are to each other, acting like a bell curve centered around each point. It essentially creates "peaks" around points of one class, allowing the SVM to draw circles/contours around them.
6. **Explain the role of the `gamma` parameter when using an RBF kernel.**
   *Hint:* `gamma` controls the "reach" of a single training example. High gamma = short reach (bumpy boundary, high variance/overfitting). Low gamma = far reach (smooth boundary, high bias/underfitting).

### Practical Questions
7. **If you have a dataset with 5 million rows, would you use `SVC(kernel='rbf')`?**
   *Hint:* No. Training time for non-linear SVMs scales terribly (between $O(n^2)$ and $O(n^3)$). It would take forever.
8. **Why must you scale your data before feeding it to an SVM?**
   *Hint:* SVMs maximize the physical distance between data points. If one feature is measured in thousands and another in decimals, the algorithm will completely ignore the decimal feature.
9. **How do you get probability estimates from an SVM?**
   *Hint:* You must pass `probability=True` when initializing the model. However, this slows down training significantly because it runs an internal 5-fold cross-validation (Platt scaling) to calibrate the probabilities.

### Output Interpretation
10. **You train an SVM with `C=1000` and `gamma=100`. Your training accuracy is 100%, but your validation accuracy is 50%. What happened?**
    *Hint:* You heavily overfit the data. `C=1000` forces a hard margin, and `gamma=100` makes the boundary hyper-sensitive to every single point. You should lower both values.

---

## Python Code Examples

### `example-01-basic-svm-linear.py`

```python
"""
Example 01: Basic Linear Support Vector Machine
Goal: Understand margins, support vectors, and the C parameter on linearly separable data.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Create a synthetic, perfectly separable dataset
# Class 0: x and y are small
# Class 1: x and y are large
X = np.array([
    [1, 1], [2, 2], [1.5, 1.8], [2, 1], # Class 0
    [8, 8], [9, 9], [8.5, 8.8], [9, 8]  # Class 1
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# 2. Train a Linear SVM with a "Hard" Margin (High C)
# C=1000 tells the model: "Do not allow any misclassifications"
model_hard = SVC(kernel='linear', C=1000)
model_hard.fit(X, y)

print("--- Hard Margin SVM ---")
print(f"Accuracy: {accuracy_score(y, model_hard.predict(X)) * 100}%")
print(f"Number of Support Vectors: {model_hard.n_support_} (Total: {sum(model_hard.n_support_)})")
print("The support vectors are the specific points closest to the boundary.")
print(f"Support Vector Coordinates:\n{model_hard.support_vectors_}")

# 3. Add an "Outlier" that messes up the clean separation
X_noisy = np.vstack((X, [7, 2])) # A Class 1 point mixed in with Class 0
y_noisy = np.append(y, 1)

# 4. Train a Linear SVM with a "Soft" Margin (Low C)
# C=0.01 tells the model: "It's okay to make a mistake if it means a better overall boundary"
model_soft = SVC(kernel='linear', C=0.01)
model_soft.fit(X_noisy, y_noisy)

print("\n--- Soft Margin SVM (with noisy data) ---")
print(f"Accuracy: {accuracy_score(y_noisy, model_soft.predict(X_noisy)) * 100}%")
print(f"Number of Support Vectors: {model_soft.n_support_} (Total: {sum(model_soft.n_support_)})")
print("Notice how there are more support vectors now. The margin is wider, so more points fall inside it or cross it.")
```

### `example-02-svm-kernels.py`

```python
"""
Example 02: The Kernel Trick
Goal: Show how non-linear kernels (like RBF) can separate data that a linear line cannot.
"""

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Generate "Circles" data
# This creates a dataset where Class 1 is a small circle inside a larger Class 0 circle.
# A straight line CANNOT separate this.
X, y = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("--- Data Generation ---")
print("Data consists of an inner circle and an outer circle.")

# 2. Try to fit a Linear SVM
linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train, y_train)
acc_linear = accuracy_score(y_test, linear_svm.predict(X_test))

print("\n--- Linear Kernel ---")
print(f"Accuracy: {acc_linear * 100:.2f}%")
print("As expected, a straight line fails completely (approx 50% accuracy).")

# 3. Try to fit an RBF (Radial Basis Function) SVM
rbf_svm = SVC(kernel='rbf', C=1.0, gamma='scale')
rbf_svm.fit(X_train, y_train)
acc_rbf = accuracy_score(y_test, rbf_svm.predict(X_test))

print("\n--- RBF Kernel ---")
print(f"Accuracy: {acc_rbf * 100:.2f}%")
print("The RBF kernel projects the data into a higher dimension where a flat plane CAN separate it.")
print("When projected back to 2D, the boundary looks like a circle!")
```

### `example-03-real-world-medical.py`

```python
"""
Example 03: Real-World Scenario - Medical Diagnosis
Goal: Apply SVM to a high-dimensional tabular dataset, utilizing scaling and probability outputs.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. Load Breast Cancer Dataset (30 numerical features)
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target  # 0 = Malignant, 1 = Benign

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. CRITICAL: Scale the data
# SVMs calculate distance. Without scaling, features with large ranges dominate.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Initialize Model with probability=True
# probability=True makes training slower, but allows us to use .predict_proba() later
model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)

# 4. Train
model.fit(X_train_scaled, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_scaled)
print("--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# 6. View Probabilities
probs = model.predict_proba(X_test_scaled)
print("\n--- Probability Output for First 3 Patients ---")
for i in range(3):
    pred_class = cancer.target_names[y_pred[i]]
    prob_benign = probs[i][1] * 100
    prob_malignant = probs[i][0] * 100
    print(f"Patient {i+1}: Predicted {pred_class}")
    print(f"  Confidence -> Benign: {prob_benign:.1f}% | Malignant: {prob_malignant:.1f}%")
```
