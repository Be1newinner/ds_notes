# Practice Exercises: Support Vector Machines

These exercises are designed to test your conceptual understanding and coding skills.

## Conceptual Questions
1. Why does an SVM only care about the Support Vectors? If you deleted 90% of the training data that are far away from the margin, would the model change?
2. You train an SVM with an RBF kernel and achieve 100% training accuracy but 60% test accuracy. The model is overfitting. Which two hyperparameters (`C` and `gamma`) should you decrease/increase to simplify the model?
3. What is the main disadvantage of setting `probability=True` in `SVC`?

## Coding Tasks

### Task 1: Linear vs RBF
1. Use `sklearn.datasets.make_moons(n_samples=200, noise=0.15)` to generate a non-linear dataset.
2. Scale the data.
3. Train two models: `SVC(kernel='linear')` and `SVC(kernel='rbf')`.
4. Compare their accuracy.

### Task 2: The Effect of Gamma
Using the `make_moons` dataset from Task 1:
1. Train three different `SVC(kernel='rbf')` models with `gamma` set to `0.1`, `1.0`, and `10.0`. Keep `C=1.0`.
2. Look at the training accuracy vs testing accuracy for each. 
3. Based on the accuracy scores, what happens as `gamma` gets larger?

### Task 3: LinearSVC for Speed
1. Generate a large, simple dataset using `sklearn.datasets.make_classification(n_samples=50000, n_features=20)`.
2. Time how long it takes to train `SVC(kernel='linear')` using the `time` module.
3. Time how long it takes to train `LinearSVC()`.
4. Compare the training times and their accuracies.
