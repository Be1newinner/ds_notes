# Interview Questions: Data Splitting

## Beginner Questions
1. **What is the difference between a training set and a testing set?**
   - *Answer*: The training set is used to teach the model patterns in the data. The testing set is kept completely hidden from the model during training and is only used at the end to evaluate how well the model generalizes to new, unseen data.
2. **Why can't we just test the model on the same data we used to train it?**
   - *Answer*: Because the model might simply memorize the training data (overfitting). Testing on training data gives a falsely high accuracy score that will not hold up in production.

## Conceptual Questions
3. **What is Cross-Validation and why is it preferred over a simple train-test split?**
   - *Answer*: Cross-validation involves splitting the data into 'k' folds and training/testing the model 'k' times, rotating the test fold each time. It is preferred because a single train-test split can be biased by a "lucky" or "unlucky" random shuffle. CV gives a more reliable average performance metric.
4. **Explain what Data Leakage is in the context of data splitting.**
   - *Answer*: Data leakage occurs when information from outside the training dataset (i.e., from the test set) is used to create the model. A classic example is scaling/normalizing the *entire* dataset before splitting it, meaning the training data was influenced by the test data's mean/variance.
5. **What is stratification? When is it absolutely necessary?**
   - *Answer*: Stratification ensures that the train and test sets have the exact same proportion of class labels as the original dataset. It is absolutely necessary when dealing with highly imbalanced datasets (e.g., predicting a rare disease), ensuring the test set actually contains examples of the minority class.

## Practical / Scenario Questions
6. **You are building a time-series model to predict stock prices. How do you split your data?**
   - *Answer*: You **cannot** use a random `train_test_split`. If you randomly shuffle time-series data, you are letting the model "look into the future" to predict the past. You must use a chronological split (e.g., train on 2018-2022, test on 2023).
7. **You have a dataset of 10,000 patient X-rays. Some patients have 5 X-rays, some have 1. How do you split the data?**
   - *Answer*: You must use a "Group" split (like `GroupKFold`). You have to ensure that all X-rays from the *same* patient go entirely into either the train set or the test set. If Patient A's images are in both, the model might just learn to recognize Patient A's bone structure, not the actual disease.
