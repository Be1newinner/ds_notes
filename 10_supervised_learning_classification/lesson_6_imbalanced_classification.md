# Imbalanced Classification

## Learning Objective
Understand why standard classification models fail when one class significantly outnumbers another, why accuracy is a misleading metric, and how to use data-level and algorithmic-level techniques to solve this problem.

## What Is This Topic?
An "imbalanced dataset" occurs when the classes you are trying to predict are not represented equally. For example, in credit card fraud detection, 99.9% of transactions are legitimate (Class 0), and only 0.1% are fraudulent (Class 1). 

## Why This Topic Matters
Most real-world business problems are inherently imbalanced. Customers rarely click ads, machines rarely break down, and patients rarely have rare diseases. If you don't know how to handle imbalanced data, your models will be completely useless in the real world.

## The Core Problem: The "Accuracy Paradox"
If 99% of your data is Class 0, a "dumb" model that *always* predicts Class 0 will achieve 99% accuracy. 
To a beginner, 99% accuracy sounds amazing. To a business, the model is completely worthless because it completely failed to detect the 1% of cases that actually matter (the fraud).

## How to Fix It

There are generally three ways to handle imbalanced data:

### 1. Change the Metrics
Stop using Accuracy. Use metrics that focus on the minority class:
- **Precision**: Out of all the ones the model predicted as Fraud, how many were *actually* Fraud?
- **Recall (Sensitivity)**: Out of all the *actual* Fraud cases, how many did the model successfully find?
- **F1-Score**: The harmonic mean of Precision and Recall.

### 2. Algorithmic Changes (Class Weights)
Most Scikit-Learn models have a parameter called `class_weight`. By setting it to `'balanced'`, you tell the algorithm to pay *more attention* to the minority class. Mathematically, it applies a heavier penalty when the model gets a minority class prediction wrong.

### 3. Data-Level Changes (Resampling)
Change the data itself *before* feeding it to the model.
- **Undersampling**: Randomly delete rows from the majority class until the classes are equal. (Warning: You lose a lot of data).
- **Oversampling (SMOTE)**: Create fake, synthetic examples of the minority class to balance the dataset.

## What is SMOTE?
**S**ynthetic **M**inority **O**versampling **TE**chnique. 
Instead of just duplicating existing fraud rows, SMOTE looks at a fraud row, finds its nearest fraud neighbors, and generates a brand new, slightly different fraud row somewhere in between them.

## Golden Rule of Resampling
**NEVER resample your test set.** 
You only undersample or oversample the *training* data. The test data must remain a true, imbalanced representation of the real world, or else your evaluation metrics will be a lie.

## Code References
- `code/example-01-class-weights.py`
- `code/example-02-smote.py`
- `code/example-03-real-world-credit-card.py`


---

## Method Options: Imbalanced Classification

This document covers the `imbalanced-learn` library and `class_weight` parameters.

### 1. Algorithmic Approach: `class_weight`

Most models in Scikit-Learn (LogisticRegression, RandomForestClassifier, SVC, DecisionTreeClassifier) accept a `class_weight` argument.

#### Syntax
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight='balanced')
```

#### Options
- **`None`** (default): All classes carry weight one.
- **`'balanced'`**: Automatically adjusts weights inversely proportional to class frequencies in the input data: `n_samples / (n_classes * np.bincount(y))`.
- **Custom Dictionary**: You can manually pass a dictionary: `class_weight={0: 1, 1: 10}` means Class 1 is penalized 10 times harder than Class 0 for a mistake.

---

### 2. Data Approach: `imbalanced-learn` (imblearn)

`imblearn` is a library built specifically to work with scikit-learn for handling imbalanced data.

#### Oversampling with SMOTE
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

#### Undersampling with RandomUnderSampler
```python
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
```

#### The `imblearn` Pipeline
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

---

## Examples: Imbalanced Classification

Here is a breakdown of the Python examples provided in the `code/` directory.

### 1. Fixing with Class Weights (`example-01-class-weights.py`)
- **Goal:** Show how easily a model fails on imbalanced data, and how a single parameter change can improve it.
- **Dataset:** A 95% / 5% synthetic dataset.
- **Key Concepts Shown:** 
  - Training a standard model and looking at the terrible Recall for the minority class.
  - Retraining the exact same model with `class_weight='balanced'`.
  - Comparing the classification reports.
- **Takeaway:** `class_weight='balanced'` should be your immediate first step anytime you have an imbalanced dataset. It requires zero data manipulation.

### 2. Fixing with SMOTE (`example-02-smote.py`)
- **Goal:** Learn how to synthetically generate data to balance the classes.
- **Dataset:** A 90% / 10% synthetic dataset.
- **Key Concepts Shown:** 
  - Using `imblearn.over_sampling.SMOTE`.
  - The Golden Rule: Only applying SMOTE to `X_train`, leaving `X_test` alone.
  - Printing the counts of classes before and after SMOTE.
- **Takeaway:** SMOTE physically changes the data size by generating new points, which can help models (especially distance-based ones) learn the minority class better.

### 3. Real-World Scenario: Credit Card Fraud (`example-03-real-world-credit-card.py`)
- **Goal:** Put everything together on a highly realistic, extremely imbalanced dataset.
- **Dataset:** Simulated credit card data with a 99% / 1% imbalance.
- **Key Concepts Shown:** 
  - The "Accuracy Paradox" (showing that predicting '0' every time gets 99% accuracy).
  - Comparing a baseline Random Forest vs. a Random Forest with SMOTE.
  - Focusing evaluation strictly on Precision, Recall, and the F1-Score.
- **Takeaway:** In business, deciding between SMOTE and Class Weights depends on whether you care more about Precision (preventing false alarms) or Recall (catching every single fraudster).

---

## Practice Exercises: Imbalanced Classification

These exercises are designed to test your conceptual understanding and coding skills.

### Conceptual Questions
1. Why is Accuracy a terrible metric to use when trying to predict if a patient has a rare disease (which occurs in 1 out of 10,000 people)?
2. If your model has High Precision but Low Recall, what does that mean in plain English? Provide an example where this is acceptable.
3. You apply SMOTE to your entire dataset (X), and then you do `train_test_split`. You get a 99% F1-score on the test set. Why is your result completely invalid?

### Coding Tasks

#### Task 1: The Accuracy Paradox
1. Use `sklearn.datasets.make_classification` with `weights=[0.99, 0.01]` to generate an imbalanced dataset of 5000 rows.
2. Write a dummy python function that completely ignores the input data and just returns an array of all zeros.
3. Calculate the accuracy of your dummy function. Calculate the Recall and F1-score. 
4. Print the results to prove how misleading accuracy is.

#### Task 2: Class Weight vs SMOTE
Using the dataset from Task 1:
1. Train a `LogisticRegression` model with `class_weight=None`. Print the classification report.
2. Train a `LogisticRegression` model with `class_weight='balanced'`. Print the classification report.
3. Use SMOTE to resample the training data. Train a `LogisticRegression` model with `class_weight=None` on the SMOTE data. Print the classification report.
4. Compare the Recall of class 1 across all three methods.

#### Task 3: Adjusting the Threshold Manually
Sometimes you don't need SMOTE or class weights; you just need to change the decision threshold.
1. Train a standard `RandomForestClassifier` on an imbalanced dataset.
2. Use `.predict_proba()` to get the raw probabilities.
3. By default, the threshold is 0.5. Write code to classify a row as Class 1 if the probability is $\ge 0.15$ (a much lower threshold).
4. Evaluate how this changes your Precision and Recall.

---

## Interview Questions: Imbalanced Classification

### Beginner Questions
1. **What does it mean if a dataset is "imbalanced"?**
   *Hint:* It means the target variable has a very uneven distribution of classes (e.g., 99% Class 0, 1% Class 1).
2. **Why shouldn't you use Accuracy to evaluate an imbalanced model?**
   *Hint:* Because a model can achieve very high accuracy simply by predicting the majority class every single time, while completely failing to identify the minority class.
3. **What is SMOTE?**
   *Hint:* Synthetic Minority Over-sampling Technique. It is an algorithm that creates synthetic, artificial rows of data for the minority class to balance the dataset.

### Conceptual Questions
4. **Explain the difference between Precision and Recall.**
   *Hint:* Precision asks: "Of all the times the model cried 'Wolf!', how many times was there actually a wolf?" Recall asks: "Out of all the actual wolves out there, how many did the model find?"
5. **How does the `class_weight='balanced'` parameter actually work under the hood?**
   *Hint:* It modifies the loss function of the algorithm. Normally, getting a Class 0 prediction wrong and a Class 1 prediction wrong incur the same penalty. `class_weight='balanced'` applies a massive mathematical penalty if the model gets the rare minority class wrong, forcing the model to care about it.
6. **If you have a strict budget and investigating false alarms costs you $10,000 each, do you want to optimize for Precision or Recall?**
   *Hint:* Precision. You want to make absolutely sure that when your model flags something, it is actually correct, to avoid wasting money on false alarms.

### Practical Questions
7. **Explain the "Golden Rule" of applying SMOTE.**
   *Hint:* You must NEVER apply SMOTE to the validation or test sets. You split your data *first*, apply SMOTE *only* to the training data, and then evaluate on the pristine, untouched test data.
8. **What is Data Leakage in the context of SMOTE and Cross-Validation?**
   *Hint:* If you apply SMOTE to the entire dataset *before* running K-Fold Cross Validation, the synthetic data generated using information from the test folds will leak into the training folds, resulting in massively inflated and fake evaluation scores.
9. **Instead of algorithmic weights or SMOTE, how can you solve an imbalance problem using just the `predict_proba()` method?**
   *Hint:* By default, models predict Class 1 if the probability is > 0.5. You can manually lower this decision threshold (e.g., to 0.1) so the model becomes much more aggressive in predicting the minority class.

### Output Interpretation
10. **Your model has a Precision of 0.99 and a Recall of 0.05. Is this a good model for detecting terminal cancer?**
    *Hint:* No, it is terrible. It means that when the model says you have cancer, it is almost certainly right (High Precision). However, it misses 95% of the actual cancer cases (Low Recall). For medical diagnoses, you generally want High Recall to ensure no sick patients slip through the cracks.

---

## Python Code Examples

### `example-01-class-weights.py`

```python
"""
Example 01: Fixing Imbalance with Class Weights
Goal: Show how class_weight='balanced' changes the focus of the model without altering data.
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Create highly imbalanced data (95% Class 0, 5% Class 1)
X, y = make_classification(n_samples=2000, n_features=10, weights=[0.95], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training Class 0 count: {sum(y_train == 0)}")
print(f"Training Class 1 count: {sum(y_train == 1)}")

# --- EXPERIMENT 1: Standard Model ---
model_standard = LogisticRegression(random_state=42)
model_standard.fit(X_train, y_train)
y_pred_std = model_standard.predict(X_test)

print("\n--- Standard Logistic Regression ---")
print(classification_report(y_test, y_pred_std))
print("Notice how the Recall for Class 1 is very low. The model ignores it to keep high overall accuracy.")

# --- EXPERIMENT 2: Balanced Model ---
# We tell the algorithm to penalize Class 1 mistakes heavily
model_balanced = LogisticRegression(class_weight='balanced', random_state=42)
model_balanced.fit(X_train, y_train)
y_pred_bal = model_balanced.predict(X_test)

print("\n--- Balanced Logistic Regression ---")
print(classification_report(y_test, y_pred_bal))
print("Notice how Recall for Class 1 jumped significantly!")
print("Trade-off: We sacrificed some Precision (more false alarms) to catch the minority class.")
```

### `example-02-smote.py`

```python
"""
Example 02: Fixing Imbalance with SMOTE
Goal: Learn how to generate synthetic data properly.
Note: You must have 'imbalanced-learn' installed (pip install imbalanced-learn)
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# Import SMOTE
from imblearn.over_sampling import SMOTE

# 1. Create imbalanced data (90% Class 0, 10% Class 1)
X, y = make_classification(n_samples=1000, n_features=5, weights=[0.90], random_state=42)

# 2. Split the data FIRST (The Golden Rule)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("--- BEFORE SMOTE ---")
print(f"Train Class 0 count: {sum(y_train == 0)}")
print(f"Train Class 1 count: {sum(y_train == 1)}")

# 3. Apply SMOTE to the TRAINING data ONLY
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\n--- AFTER SMOTE ---")
print(f"Train Class 0 count: {sum(y_train_resampled == 0)}")
print(f"Train Class 1 count: {sum(y_train_resampled == 1)}")
print("SMOTE generated synthetic rows so the classes are perfectly equal 50/50.")

# 4. Train model on the resampled data
model = RandomForestClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# 5. Evaluate on the ORIGINAL, UNTOUCHED test data
y_pred = model.predict(X_test)

print("\n--- Classification Report (Evaluated on Untouched Test Set) ---")
print(classification_report(y_test, y_pred))
print("By feeding the model balanced data during training, it learned to recognize Class 1 better.")
```

### `example-03-real-world-credit-card.py`

```python
"""
Example 03: Real-World Scenario - Credit Card Fraud
Goal: Compare different techniques on a highly imbalanced dataset and interpret business metrics.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE

# 1. Simulate extremely imbalanced Credit Card Data (99% / 1%)
np.random.seed(42)
n_samples = 10000

amount = np.random.exponential(scale=50, size=n_samples)
time_since_last_txn = np.random.exponential(scale=10, size=n_samples)

# Very strict logic for Fraud (Class 1)
log_odds = 0.05 * amount - 0.1 * time_since_last_txn - 6
prob_fraud = 1 / (1 + np.exp(-log_odds))
fraud = (np.random.rand(n_samples) < prob_fraud).astype(int)

X = pd.DataFrame({'Amount': amount, 'Time_Since_Last': time_since_last_txn})
y = fraud

print(f"Total Legit (0): {sum(y==0)} | Total Fraud (1): {sum(y==1)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Approach 1: Ignorance (Do Nothing) ---
rf_baseline = RandomForestClassifier(random_state=42)
rf_baseline.fit(X_train, y_train)
pred_baseline = rf_baseline.predict(X_test)
print("\n--- Baseline (No Adjustments) ---")
print(classification_report(y_test, pred_baseline, zero_division=0))

# --- Approach 2: Class Weights ---
rf_weighted = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_weighted.fit(X_train, y_train)
pred_weighted = rf_weighted.predict(X_test)
print("\n--- With Class Weights ---")
print(classification_report(y_test, pred_weighted))

# --- Approach 3: SMOTE ---
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_sm, y_train_sm)
pred_smote = rf_smote.predict(X_test)
print("\n--- With SMOTE ---")
print(classification_report(y_test, pred_smote))

print("\nBusiness Conclusion:")
print("The baseline model missed almost all the fraud. Class Weights improved Recall drastically.")
print("Depending on the algorithm, SMOTE can sometimes outperform Class Weights by providing")
print("physical data points for the trees to split on.")
```
