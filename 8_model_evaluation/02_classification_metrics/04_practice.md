# Practice Exercises: Classification Metrics

## Exercise 1: Manual Confusion Matrix
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

## Exercise 2: The Accuracy Trap
1. Create dummy lists in Python:
   `y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]`
   `y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`
2. Calculate the `accuracy_score`.
3. Calculate the `recall_score`.
4. Explain why the model is terrible despite the high accuracy.

## Exercise 3: Optimizing the Business Case
For each scenario below, state whether you should optimize for **Precision** or **Recall**, and explain why.
1. A fingerprint scanner to unlock a high-security bank vault.
2. An AI reading X-rays to detect early signs of a brain tumor.
3. A system automatically deleting emails it thinks are phishing scams.
4. A factory robot throwing away products that it thinks have manufacturing defects (but the products are very cheap to make).

## Exercise 4: Code Implementation
1. Load the `breast_cancer` dataset from `sklearn.datasets`.
2. Perform a train/test split.
3. Train a `LogisticRegression` model.
4. Predict on the test set.
5. Print the full `classification_report`.
6. Based on the report, is the model better at identifying Malignant or Benign tumors?
