# Examples: Logistic Regression

Here is a breakdown of the Python examples provided in the `code/` directory.

## 1. Basic Binary Classification (`example-01-basic-logistic.py`)
- **Goal:** Introduce the fundamental usage of Logistic Regression.
- **Dataset:** A synthetic dataset representing a simple pass/fail scenario based on study hours and sleep.
- **Key Concepts Shown:** Model initialization, `fit()`, `predict()`, and `predict_proba()`.
- **Takeaway:** Shows how Logistic Regression outputs a probability and converts it into a hard 0 or 1 class prediction based on a 0.5 threshold.

## 2. Multiclass Classification (`example-02-multiclass.py`)
- **Goal:** Show how Logistic Regression handles more than two categories.
- **Dataset:** The classic Iris dataset (predicting 3 different flower species).
- **Key Concepts Shown:** `multi_class='multinomial'`, interpreting the shape of `predict_proba()` (which now outputs 3 probabilities that sum to 1), and `coef_` shapes.
- **Takeaway:** Logistic Regression is not limited to binary (Yes/No) questions.

## 3. Real-World Scenario: Customer Churn (`example-03-real-world-churn.py`)
- **Goal:** Apply Logistic Regression to a realistic business problem.
- **Dataset:** A simulated telecom customer dataset containing demographics, account info, and whether they canceled their subscription (Churn).
- **Key Concepts Shown:** 
  - Data scaling (StandardScaler) which is critical for regularized logistic regression.
  - Interpreting coefficients to drive business insights (e.g., "High monthly charges strongly increase the chance of churn").
  - Evaluating using a confusion matrix and classification report instead of just simple accuracy.
- **Takeaway:** How to translate mathematical model outputs into actionable business advice.
