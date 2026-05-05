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
