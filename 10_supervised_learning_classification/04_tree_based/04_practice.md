# Practice Exercises: Tree-Based Classification

These exercises are designed to test your conceptual understanding and coding skills.

## Conceptual Questions
1. Why is feature scaling (like `StandardScaler`) completely unnecessary for Decision Trees and Random Forests?
2. Explain the difference between Bagging (Random Forest) and Boosting (Gradient Boosting).
3. If your Random Forest model is overfitting, what is the best hyperparameter to adjust to fix it? (Hint: Think about the depth of the individual trees).

## Coding Tasks

### Task 1: Overfitting a Decision Tree
1. Load the `digits` dataset from `sklearn.datasets`.
2. Split into 80/20 train/test.
3. Train a `DecisionTreeClassifier` with absolutely no parameters (default).
4. Print the accuracy on the *training* set, and then the accuracy on the *test* set. What do you observe?
5. Train a new tree, but set `max_depth=5`. Compare the train and test accuracies again.

### Task 2: The Power of the Forest
Using the same `digits` dataset:
1. Train a `RandomForestClassifier` with `n_estimators=10` and print the test accuracy.
2. Train another with `n_estimators=100` and print the test accuracy.
3. Did the accuracy improve? Did it take noticeably longer to train?

### Task 3: Extracting Feature Importance
1. Load the `breast_cancer` dataset (`load_breast_cancer`).
2. Train a `RandomForestClassifier`.
3. Extract the `feature_importances_`.
4. Write a script to find and print the name of the **single most important feature** in determining whether a tumor is malignant or benign according to the model.
