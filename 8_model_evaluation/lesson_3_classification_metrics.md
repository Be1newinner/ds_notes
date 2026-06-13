# Classification Metrics

## Learning Objective
Students will learn how to evaluate models that predict categories (classes). They will learn why simple "Accuracy" is often misleading and how to use the Confusion Matrix to derive Precision, Recall, and the F1-Score to make better business decisions.

## What Is This Topic?
When a model predicts a category (e.g., "Spam" vs "Not Spam", or "Disease" vs "No Disease"), we need to count how many times it was right and how many times it made a mistake. Classification metrics are the specific formulas we use to calculate the severity and type of those mistakes.

## Why This Topic Matters
Accuracy is a dangerous metric. Imagine a dataset of 100 emails, where 99 are normal and 1 is spam. A broken model that simply predicts "Normal" every single time will be 99% accurate! But it is completely useless because it failed to catch the 1 spam email. We need deeper metrics to understand model behavior.

## Core Intuition: The Boy Who Cried Wolf
- **True Positive (TP)**: The boy cries "Wolf!", and there *is* a wolf. (Good)
- **True Negative (TN)**: The boy is quiet, and there is *no* wolf. (Good)
- **False Positive (FP)**: The boy cries "Wolf!", but there is *no* wolf. (False Alarm / Type I Error)
- **False Negative (FN)**: The boy is quiet, but there *is* a wolf. (Disaster / Type II Error)

## Key Concepts
### 1. The Confusion Matrix
A 2x2 table that explicitly shows the TP, TN, FP, and FN counts. It is the foundation of all other metrics.

### 2. Accuracy
Total correct predictions divided by total predictions. (Only useful if classes are perfectly balanced).
`Accuracy = (TP + TN) / Total`

### 3. Precision (Quality)
Out of all the times the model *said* "Positive", how many were actually Positive?
`Precision = TP / (TP + FP)`
**Focus**: Minimizing False Positives (False Alarms).

### 4. Recall / Sensitivity (Quantity)
Out of all the *actual* Positives in reality, how many did the model manage to find?
`Recall = TP / (TP + FN)`
**Focus**: Minimizing False Negatives (Missed Threats).

### 5. F1-Score
The harmonic mean of Precision and Recall. It gives a single score that balances both. If either Precision or Recall is very low, the F1 score will be very low.

### 6. ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
A graph showing the trade-off between the True Positive Rate and the False Positive Rate at different threshold levels. An AUC of 1.0 is perfect, 0.5 is random guessing.

## Output / Result Interpretation
- **High Precision, Low Recall**: The model is extremely cautious. It only flags something if it is 100% sure. It misses a lot of targets, but when it fires, it is right. (Good for YouTube video recommendations).
- **High Recall, Low Precision**: The model is hyper-sensitive. It flags everything that looks remotely suspicious. It catches all the targets, but annoys you with many false alarms. (Good for Airport Security).

## Real-World Uses
- **Medical Diagnostics (Optimize for Recall)**: It's better to accidentally tell a healthy person they *might* have cancer (False Positive) and run more tests, than to tell a sick person they are healthy (False Negative) and send them home to die.
- **Spam Filter (Optimize for Precision)**: It's better to let a spam email into the inbox (False Negative) than to accidentally send an important job offer to the Spam folder (False Positive).

## Common Mistakes
- Relying solely on Accuracy for imbalanced datasets.
- Not understanding the business cost of a False Positive versus a False Negative before choosing a metric.

## Related Methods
- **Classification Report**: A scikit-learn function that prints Precision, Recall, and F1 for *all* classes at once.
- **Precision-Recall Curve**: Used instead of ROC for highly imbalanced datasets.

## Code References
- `code/example-01-accuracy-confusion-matrix.py`
- `code/example-02-precision-recall-f1.py`
- `code/example-03-roc-auc.py`
- `code/example-04-classification-report.py`


---

## Method & Options: Classification Metrics

This document details the common scikit-learn functions used to evaluate classification models. All these functions are found in `sklearn.metrics`.

### 1. `confusion_matrix`

#### Purpose
Calculates the raw counts of True Negatives, False Positives, False Negatives, and True Positives.

#### Syntax
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
```

#### Output Interpretation
Returns a 2D array (list of lists). For binary classification:
`[[TN, FP],`
 `[FN, TP]]`
To easily visualize this, you usually pass the `cm` to `sns.heatmap()` or use `ConfusionMatrixDisplay`.

---

### 2. `accuracy_score`, `precision_score`, `recall_score`, `f1_score`

#### Purpose
Calculates the specific metric as a single float number between 0.0 and 1.0.

#### Syntax
```python
from sklearn.metrics import precision_score, recall_score
p = precision_score(y_true, y_pred)
r = recall_score(y_true, y_pred)
```

#### Common Arguments
- `y_true`: The actual ground truth labels.
- `y_pred`: The labels predicted by your model.
- `pos_label` (int/str): Which class should be considered the "Positive" class (default is `1`).
- `average` (string): Vital for multi-class problems (e.g., predicting 3 or more categories).
  - `'binary'`: (Default) Only reports results for the class specified by `pos_label`.
  - `'macro'`: Calculates metrics for each class individually, then takes the unweighted mean. Does not take class imbalance into account.
  - `'weighted'`: Calculates metrics for each class, then takes the average weighted by the number of true instances for each class.

---

### 3. `classification_report`

#### Purpose
The ultimate time-saver. It builds a text report showing the main classification metrics (precision, recall, f1, support) for every single class in your dataset.

#### Syntax
```python
from sklearn.metrics import classification_report
report = classification_report(y_true, y_pred, target_names=["Not Spam", "Spam"])
print(report)
```

#### Output Interpretation
- **support**: The actual number of occurrences of that class in the `y_true` dataset.
- **macro avg**: The standard average of the metrics across all classes.
- **weighted avg**: The average weighted by the `support` (the size of each class).

---

### 4. `roc_auc_score`

#### Purpose
Calculates the Area Under the Receiver Operating Characteristic Curve. This metric evaluates how well the model can distinguish between the classes at various threshold levels.

#### Syntax
```python
from sklearn.metrics import roc_auc_score
# NOTE: You MUST pass probabilities, not just the final 0/1 predictions!
y_pred_proba = model.predict_proba(X_test)[:, 1] 
auc = roc_auc_score(y_true, y_pred_proba)
```

#### Important Rule
Unlike `accuracy` or `precision`, `roc_auc_score` should be fed the *probabilities* outputted by the model (`predict_proba()`), not the hard class predictions (`predict()`). You want the probability of the positive class (usually column index `1`).

---

## Classification Metrics Examples

This document explains the python examples provided in the `code/` directory.

### 1. Accuracy and the Confusion Matrix (`example-01-accuracy-confusion-matrix.py`)
This script demonstrates the foundation of classification evaluation.
- It generates a dataset and trains a basic Logistic Regression model.
- It calculates the Accuracy score.
- Most importantly, it generates a Confusion Matrix and prints it cleanly, showing exactly *where* the model is making its mistakes (False Positives vs False Negatives).

### 2. Precision, Recall, and F1 (`example-02-precision-recall-f1.py`)
This script explains the trade-off between Quality and Quantity.
- It creates a heavily imbalanced "Fraud Detection" dataset.
- It shows how Accuracy is deceptively high (95%).
- It calculates Precision and Recall to reveal the model is actually doing a poor job of catching the actual fraud.
- It combines them into the F1-score to give a single honest assessment.

### 3. The ROC-AUC Score (`example-03-roc-auc.py`)
This script demonstrates how to evaluate a model independent of its threshold.
- It uses the `predict_proba()` method to get the raw probability (e.g., "There is a 72% chance this is class 1").
- It calculates the ROC-AUC score.
- It explains that an AUC of 0.5 is useless (random), and 1.0 is perfect.

### 4. The Classification Report (`example-04-classification-report.py`)
This script shows the most practical, everyday tool for evaluating a classifier.
- It uses the `classification_report` function.
- It shows how to instantly get Precision, Recall, and F1 for *all* classes simultaneously without having to write separate formulas.

---

## Practice Exercises: Classification Metrics

### Exercise 1: Manual Confusion Matrix
Assume you built a model to predict if a picture is a Cat (Positive) or a Dog (Negative).
Out of 100 pictures:
- The model correctly identified 40 Cats.
- The model correctly identified 45 Dogs.
- The model said 10 Dogs were Cats.
- The model said 5 Cats were Dogs.

**Task:**
1. Draw the Confusion Matrix on paper.
2. What is the True Positive (TP) count?
3. What is the False Positive (FP) count?
4. What is the False Negative (FN) count?

### Exercise 2: The Accuracy Trap
1. Create dummy lists in Python:
   `y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]`
   `y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`
2. Calculate the `accuracy_score`.
3. Calculate the `recall_score`.
4. Explain why the model is terrible despite the high accuracy.

### Exercise 3: Optimizing the Business Case
For each scenario below, state whether you should optimize for **Precision** or **Recall**, and explain why.
1. A fingerprint scanner to unlock a high-security bank vault.
2. An AI reading X-rays to detect early signs of a brain tumor.
3. A system automatically deleting emails it thinks are phishing scams.
4. A factory robot throwing away products that it thinks have manufacturing defects (but the products are very cheap to make).

### Exercise 4: Code Implementation
1. Load the `breast_cancer` dataset from `sklearn.datasets`.
2. Perform a train/test split.
3. Train a `LogisticRegression` model.
4. Predict on the test set.
5. Print the full `classification_report`.
6. Based on the report, is the model better at identifying Malignant or Benign tumors?

---

## Interview Questions: Classification Metrics

### Beginner Questions
1. **If your classification model has 99% accuracy, does that mean it's ready for production?**
   - *Answer*: No. If the dataset is highly imbalanced (e.g., 99% of transactions are legitimate and 1% are fraud), a model that simply guesses "Legitimate" every single time will have 99% accuracy but will fail to catch any fraud. You must check Precision and Recall.
2. **What is the difference between a False Positive and a False Negative?**
   - *Answer*: A False Positive is a "False Alarm" (the model predicted the event would happen, but it didn't). A False Negative is a "Miss" (the model predicted the event would *not* happen, but it actually did).

### Conceptual Questions
3. **Explain Precision vs. Recall in simple terms.**
   - *Answer*: Precision is Quality: "When the model fires, how often is it right?" Recall is Quantity: "Out of all the targets that exist in reality, how many did the model manage to catch?"
4. **Why do we use the F1-Score instead of just taking the simple average of Precision and Recall?**
   - *Answer*: The F1-score uses the *harmonic mean*. The simple average is too forgiving. If a model has 100% Recall and 0% Precision, the simple average is 50%. The harmonic mean heavily penalizes extreme differences, dragging the F1-score down to near 0, which accurately reflects that the model is broken.
5. **What does the ROC curve represent, and what does the AUC number mean?**
   - *Answer*: The ROC curve plots the True Positive Rate against the False Positive Rate at every possible probability threshold. The AUC (Area Under Curve) is a single number summarizing that graph. An AUC of 1.0 means the model perfectly separates the classes. An AUC of 0.5 means the model is no better than flipping a coin.

### Practical / Scenario Questions
6. **You are building a spam filter. Do you tune the model for higher Precision or higher Recall?**
   - *Answer*: Higher Precision. In a spam filter, a False Positive means sending an important real email (like a job offer) to the spam folder, which is terrible. A False Negative means letting a spam email into the inbox, which is merely a mild annoyance.
7. **You are building an AI to detect defective parts in a car engine. Do you optimize for Precision or Recall?**
   - *Answer*: Higher Recall. A False Negative means letting a defective, dangerous engine part go into a car, which could cause a fatal crash. A False Positive means throwing away a perfectly good part, which just costs the company a little bit of money. Safety demands Recall.

---

## Python Code Examples

### `example-01-accuracy-confusion-matrix.py`

```python
"""
Example 01: Accuracy and the Confusion Matrix
This script shows how to build and interpret a confusion matrix.
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# 1. Create a synthetic dataset (Binary Classification)
# 1000 samples, roughly 50/50 split of class 0 and class 1
X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)

# 2. Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# 3. Predict
y_pred = model.predict(X_test)

# 4. Evaluate with Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {acc * 100:.1f}%")
print("-" * 30)

# 5. Build the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(f"[{cm[0][0]} (True Negatives)   |  {cm[0][1]} (False Positives)]")
print(f"[{cm[1][0]} (False Negatives)  |  {cm[1][1]} (True Positives)]")

# Interpretation:
# True Negatives (TN): Model correctly said "No"
# False Positives (FP): Model wrongly said "Yes" (False Alarm)
# False Negatives (FN): Model wrongly said "No" (Missed it)
# True Positives (TP): Model correctly said "Yes"
```

### `example-02-precision-recall-f1.py`

```python
"""
Example 02: Precision, Recall, and the F1-Score
This script shows why Accuracy fails on imbalanced datasets, and how Precision/Recall fix it.
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# 1. Create a highly imbalanced dataset (e.g., 95% Normal, 5% Fraud)
X, y = make_classification(n_samples=1000, weights=[0.95, 0.05], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Train a very simple, perhaps under-tuned model
model = LogisticRegression()
model.fit(X_train, y_train)

# 3. Predict
y_pred = model.predict(X_test)

# 4. The Accuracy Illusion
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy:  {acc * 100:.1f}%  <-- Looks amazing, right?")

# 5. The Honest Truth
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision * 100:.1f}%  <-- When it says Fraud, it is right {precision*100:.0f}% of the time.")
print(f"Recall:    {recall * 100:.1f}%  <-- But it only caught {recall*100:.0f}% of the actual Fraud cases!")
print(f"F1-Score:  {f1 * 100:.1f}%  <-- The true combined score of the model.")

# Conclusion:
# A model with 95% accuracy might only have a 25% F1-score if it is just guessing "Normal" most of the time!
```

### `example-03-roc-auc.py`

```python
"""
Example 03: The ROC-AUC Score
This script demonstrates how to evaluate a model independent of its threshold.
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 1. Create Data and Train Model
X, y = make_classification(n_samples=1000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# 2. Getting Probabilities (CRITICAL STEP)
# ROC-AUC needs the raw probability of the positive class (class 1), NOT the hard 0/1 prediction.
# predict_proba returns two columns: [Probability of 0, Probability of 1]
# We want the second column (index 1).
y_pred_probabilities = model.predict_proba(X_test)[:, 1]

# 3. Calculate AUC
auc_score = roc_auc_score(y_test, y_pred_probabilities)

print(f"ROC-AUC Score: {auc_score:.3f}")
print("Interpretation:")
print("0.500 = Random Guessing")
print("0.700 = Acceptable")
print("0.800 = Excellent")
print("1.000 = Perfect Model")
```

### `example-04-classification-report.py`

```python
"""
Example 04: The Classification Report
This is the most common tool used by Data Scientists on a daily basis.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# 1. Load a Multi-Class dataset (Iris has 3 classes of flowers)
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Train a Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 3. Predict
y_pred = model.predict(X_test)

# 4. Print the Classification Report
# We pass target_names so the report uses the real flower names instead of just 0, 1, 2
report = classification_report(y_test, y_pred, target_names=iris.target_names)

print("Classification Report:")
print(report)

# How to read this:
# Support: How many times that flower actually appeared in the test set.
# Precision/Recall for Setosa: It is perfect. It found all of them, and made no mistakes.
# Versicolor vs Virginica: The model sometimes confuses these two. Look at the slight drops in F1-score.
```
