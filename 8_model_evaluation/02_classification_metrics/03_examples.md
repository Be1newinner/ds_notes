# Classification Metrics Examples

This document explains the python examples provided in the `code/` directory.

## 1. Accuracy and the Confusion Matrix (`example-01-accuracy-confusion-matrix.py`)
This script demonstrates the foundation of classification evaluation.
- It generates a dataset and trains a basic Logistic Regression model.
- It calculates the Accuracy score.
- Most importantly, it generates a Confusion Matrix and prints it cleanly, showing exactly *where* the model is making its mistakes (False Positives vs False Negatives).

## 2. Precision, Recall, and F1 (`example-02-precision-recall-f1.py`)
This script explains the trade-off between Quality and Quantity.
- It creates a heavily imbalanced "Fraud Detection" dataset.
- It shows how Accuracy is deceptively high (95%).
- It calculates Precision and Recall to reveal the model is actually doing a poor job of catching the actual fraud.
- It combines them into the F1-score to give a single honest assessment.

## 3. The ROC-AUC Score (`example-03-roc-auc.py`)
This script demonstrates how to evaluate a model independent of its threshold.
- It uses the `predict_proba()` method to get the raw probability (e.g., "There is a 72% chance this is class 1").
- It calculates the ROC-AUC score.
- It explains that an AUC of 0.5 is useless (random), and 1.0 is perfect.

## 4. The Classification Report (`example-04-classification-report.py`)
This script shows the most practical, everyday tool for evaluating a classifier.
- It uses the `classification_report` function.
- It shows how to instantly get Precision, Recall, and F1 for *all* classes simultaneously without having to write separate formulas.
