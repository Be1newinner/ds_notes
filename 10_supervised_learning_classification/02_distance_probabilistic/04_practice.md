# Practice Exercises: KNN and Naive Bayes

These exercises are designed to test your conceptual understanding and coding skills.

## Conceptual Questions
1. If you set $K = 1$ in K-Nearest Neighbors, what happens to the training accuracy? What will likely happen to the testing accuracy? Explain why in terms of overfitting.
2. Why is Naive Bayes called "Naive"? Provide a real-world example of where its core assumption fails.
3. You have a dataset with 50 million rows. Why might KNN be a terrible choice for this problem in a real-time production environment?

## Coding Tasks

### Task 1: Finding the Best K
1. Load the `wine` dataset from `sklearn.datasets` (`load_wine()`).
2. Split the data into train and test sets (80/20).
3. Scale the features using `StandardScaler`.
4. Write a `for` loop that trains a `KNeighborsClassifier` for every odd value of $K$ from 1 to 21.
5. Store the test accuracy for each $K$ and print out which $K$ gave the best result.

### Task 2: Gaussian Naive Bayes Baseline
Using the exact same scaled `wine` dataset from Task 1:
1. Initialize a `GaussianNB` model.
2. Train it and predict on the test set.
3. Compare the accuracy of Naive Bayes to the best accuracy you got from KNN. Which one performed better?

### Task 3: Text Classification Pipeline
1. Create a tiny dataset of 5 sentences about sports (Label: 1) and 5 sentences about politics (Label: 0).
2. Use `sklearn.feature_extraction.text.CountVectorizer` to transform the text.
3. Train a `MultinomialNB` classifier.
4. Write a brand new sentence that mixes sports and politics (e.g., "The president threw a football"). Predict its class and print the `predict_proba` to see how confused the model is.
