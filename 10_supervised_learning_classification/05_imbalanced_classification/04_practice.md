# Practice Exercises: Imbalanced Classification

These exercises are designed to test your conceptual understanding and coding skills.

## Conceptual Questions
1. Why is Accuracy a terrible metric to use when trying to predict if a patient has a rare disease (which occurs in 1 out of 10,000 people)?
2. If your model has High Precision but Low Recall, what does that mean in plain English? Provide an example where this is acceptable.
3. You apply SMOTE to your entire dataset (X), and then you do `train_test_split`. You get a 99% F1-score on the test set. Why is your result completely invalid?

## Coding Tasks

### Task 1: The Accuracy Paradox
1. Use `sklearn.datasets.make_classification` with `weights=[0.99, 0.01]` to generate an imbalanced dataset of 5000 rows.
2. Write a dummy python function that completely ignores the input data and just returns an array of all zeros.
3. Calculate the accuracy of your dummy function. Calculate the Recall and F1-score. 
4. Print the results to prove how misleading accuracy is.

### Task 2: Class Weight vs SMOTE
Using the dataset from Task 1:
1. Train a `LogisticRegression` model with `class_weight=None`. Print the classification report.
2. Train a `LogisticRegression` model with `class_weight='balanced'`. Print the classification report.
3. Use SMOTE to resample the training data. Train a `LogisticRegression` model with `class_weight=None` on the SMOTE data. Print the classification report.
4. Compare the Recall of class 1 across all three methods.

### Task 3: Adjusting the Threshold Manually
Sometimes you don't need SMOTE or class weights; you just need to change the decision threshold.
1. Train a standard `RandomForestClassifier` on an imbalanced dataset.
2. Use `.predict_proba()` to get the raw probabilities.
3. By default, the threshold is 0.5. Write code to classify a row as Class 1 if the probability is $\ge 0.15$ (a much lower threshold).
4. Evaluate how this changes your Precision and Recall.
