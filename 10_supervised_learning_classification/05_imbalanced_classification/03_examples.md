# Examples: Imbalanced Classification

Here is a breakdown of the Python examples provided in the `code/` directory.

## 1. Fixing with Class Weights (`example-01-class-weights.py`)
- **Goal:** Show how easily a model fails on imbalanced data, and how a single parameter change can improve it.
- **Dataset:** A 95% / 5% synthetic dataset.
- **Key Concepts Shown:** 
  - Training a standard model and looking at the terrible Recall for the minority class.
  - Retraining the exact same model with `class_weight='balanced'`.
  - Comparing the classification reports.
- **Takeaway:** `class_weight='balanced'` should be your immediate first step anytime you have an imbalanced dataset. It requires zero data manipulation.

## 2. Fixing with SMOTE (`example-02-smote.py`)
- **Goal:** Learn how to synthetically generate data to balance the classes.
- **Dataset:** A 90% / 10% synthetic dataset.
- **Key Concepts Shown:** 
  - Using `imblearn.over_sampling.SMOTE`.
  - The Golden Rule: Only applying SMOTE to `X_train`, leaving `X_test` alone.
  - Printing the counts of classes before and after SMOTE.
- **Takeaway:** SMOTE physically changes the data size by generating new points, which can help models (especially distance-based ones) learn the minority class better.

## 3. Real-World Scenario: Credit Card Fraud (`example-03-real-world-credit-card.py`)
- **Goal:** Put everything together on a highly realistic, extremely imbalanced dataset.
- **Dataset:** Simulated credit card data with a 99% / 1% imbalance.
- **Key Concepts Shown:** 
  - The "Accuracy Paradox" (showing that predicting '0' every time gets 99% accuracy).
  - Comparing a baseline Random Forest vs. a Random Forest with SMOTE.
  - Focusing evaluation strictly on Precision, Recall, and the F1-Score.
- **Takeaway:** In business, deciding between SMOTE and Class Weights depends on whether you care more about Precision (preventing false alarms) or Recall (catching every single fraudster).
