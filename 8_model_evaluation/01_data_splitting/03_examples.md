# Data Splitting Examples

This document explains the python examples provided in the `code/` directory.

## 1. Simple Train-Test Split (`example-01-train-test-split.py`)
This script demonstrates the most basic form of model evaluation.
- It generates a dummy dataset (like predicting house prices).
- It splits the data 80% / 20%.
- It proves that the model performs perfectly on the data it memorized (train set), but slightly worse on data it has never seen (test set).

## 2. K-Fold Cross Validation (`example-02-kfold-cv.py`)
A single train-test split can be lucky or unlucky. This script demonstrates Cross-Validation.
- It splits the data into 5 chunks (folds).
- It trains the model 5 separate times, holding out a different chunk for testing each time.
- It prints the score for each fold, showing how model performance fluctuates depending on the data.
- It calculates the reliable "average" score.

## 3. Stratified Splitting (`example-03-stratified-cv.py`)
This is crucial for classification problems with imbalanced classes (e.g., 90% No Fraud, 10% Fraud).
- It generates a dataset with an intentional 9-to-1 imbalance.
- It shows how a normal random split might accidentally put 0% fraud cases in the test set.
- It uses `train_test_split(..., stratify=y)` to guarantee the 9-to-1 ratio exists perfectly in both the training set and the test set.
