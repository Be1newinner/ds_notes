# Submodule Map: Model Evaluation & Selection

This map provides the teaching structure and rationale for the subtopics in the Model Evaluation & Selection module.

## 1. 01_data_splitting
- **Why it is taught**: To prevent "data leakage" and overfitting. Students must learn that testing a model on its training data gives a false sense of security.
- **Format**: Theory & Code
- **Key Focus**: Train-Test Split (the basics), K-Fold Cross Validation (the robust method), and Stratified Splitting (handling imbalanced data).
- **Business Example**: Why evaluating an email spam filter only on emails it has already seen is a bad idea.

## 2. 02_classification_metrics
- **Why it is taught**: Classification tasks are extremely common, but the default metric ("Accuracy") is often deeply misleading in business scenarios.
- **Format**: Visual & Code heavy
- **Key Focus**: Confusion Matrix is the foundational tool. From there, derive Precision, Recall, F1-Score, and ROC-AUC.
- **Business Example**: Medical testing (minimizing False Negatives) vs. YouTube Recommendations (minimizing False Positives).

## 3. 03_regression_metrics
- **Why it is taught**: When predicting a continuous number (like price or temperature), "accuracy" doesn't exist. We must measure the *size* of our errors.
- **Format**: Math Intuition & Code
- **Key Focus**: MAE (simple interpretation), MSE/RMSE (penalizing large errors), and R-squared (explaining variance).
- **Business Example**: Predicting real estate prices and telling a client "our model is usually off by $10,000" (MAE).

## 4. 04_hyperparameter_tuning
- **Why it is taught**: Models rarely perform at their absolute best out-of-the-box. Students need to learn how to search for the best configuration mathematically.
- **Format**: Code heavy
- **Key Focus**: Grid Search (exhaustive but slow) vs. Random Search (faster, often just as good). The concept of model parameters vs. hyperparameters.
- **Business Example**: Automating the process of finding the best "settings" for a fraud detection model over the weekend instead of tuning manually.

## Recommended Order of Teaching
1. Data Splitting (Foundation)
2. Classification Metrics (Application 1)
3. Regression Metrics (Application 2)
4. Hyperparameter Tuning (Optimization)
