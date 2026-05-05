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
