# Practice Exercises: Logistic Regression

These exercises are designed to test your conceptual understanding and coding skills.

## Conceptual Questions
1. Why is the sigmoid function necessary in Logistic Regression? What would happen if we just used a straight line (Linear Regression) to predict classes?
2. You fit a logistic regression model predicting if an email is spam (1) or not (0). The coefficient for the word "lottery" is `+2.5`. What does this mean in plain English?
3. If your model outputs a probability of `0.49`, what class will `.predict()` output by default? How might you change the threshold if you want to be extremely careful about missing Spam emails?

## Coding Tasks

### Task 1: Basic Fit & Predict
Load the `breast_cancer` dataset from `sklearn.datasets`. Split it into 80% train and 20% test. Fit a Logistic Regression model and print the accuracy on the test set.

### Task 2: The Importance of Scaling
Using the same breast cancer dataset:
1. Fit a Logistic Regression model *without* scaling the data. Print the accuracy.
2. Scale the data using `StandardScaler`.
3. Fit a new Logistic Regression model on the scaled data. Print the accuracy.
4. Compare the two accuracies and explain the difference.

### Task 3: Changing the Decision Threshold
By default, `.predict()` uses a 0.5 threshold. 
1. Train a model on any binary dataset.
2. Use `.predict_proba()` to get the raw probabilities.
3. Write a small Python script to manually create predictions using a threshold of `0.8` (i.e., only predict class 1 if you are 80% sure). 
4. How does this change your false positives and false negatives?
