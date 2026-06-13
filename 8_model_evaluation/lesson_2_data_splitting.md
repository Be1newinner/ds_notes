# Data Splitting

## Learning Objective
Students should understand why we cannot train and test a model on the same data, and learn techniques for partitioning data effectively to estimate real-world model performance.

## What Is This Topic?
Data splitting is the practice of dividing your dataset into separate pieces: one piece to teach the model (Train), and another piece to test the model (Test/Validation).

## Why This Topic Matters
If a student takes an exam using the exact same questions they studied the night before, their high score doesn't mean they are smart—it just means they memorized the answers. Similarly, if a model is tested on its training data, it will look perfectly accurate, but fail miserably in the real world. We split data to prevent this "memorization" (Overfitting).

## Core Intuition
Imagine preparing for a driving test.
- **Training Set**: The practice hours you spend driving around your neighborhood.
- **Testing Set**: The actual driving test on a new route with an examiner.
If the driving test was on the exact same roads you practiced on, you might pass just by memorizing the turns, not by actually knowing how to drive.

## Key Concepts
- **Train Set**: The data the model learns from.
- **Test/Validation Set**: The data held back to evaluate the model.
- **Cross-Validation (K-Fold)**: Splitting the data multiple times in different ways to get a more reliable average score.
- **Stratification**: Ensuring the proportions of categories (e.g., 90% healthy, 10% sick) stay exactly the same in both the train and test sets.

## Step-by-Step Explanation
1. Gather your complete dataset.
2. Decide on a split ratio (commonly 80% train, 20% test).
3. Randomly shuffle the data to avoid alphabetical or chronological bias.
4. Separate the features (X) and the target variable (y).
5. Pass the data through a splitting function.
6. Train the model *only* on the Training set.
7. Predict and evaluate *only* on the Test set.

## Output / Result Interpretation
A successful data split results in 4 distinct objects:
- `X_train`: The features used for learning.
- `X_test`: The features used for testing.
- `y_train`: The true answers used for learning.
- `y_test`: The true answers used for testing.

## Real-World Uses
- **Credit Risk**: Splitting historical loan data to ensure the model can predict default on *future* applicants, not just past ones.
- **Medical Imaging**: Ensuring images from the same patient aren't accidentally put in both the train and test sets (group splitting).

## Advantages
- Provides an honest estimate of model performance.
- Essential for detecting overfitting.
- Easy to implement with standard libraries.

## Limitations
- Reduces the amount of data available for the model to actually learn from.
- A single random split might be "lucky" or "unlucky" (which is why Cross-Validation exists).

## Common Mistakes
- **Data Leakage**: Preprocessing the whole dataset (like filling missing values with the mean) *before* splitting the data. You must split first, then preprocess!
- **Not Stratifying Imbalanced Data**: If you have 1% fraud cases, a random split might result in 0 fraud cases in your test set.
- **Time Series Shuffling**: Randomly splitting time-based data (like stock prices) allows the model to predict the past using the future.

## Related Methods
- **Train/Validation/Test Split**: Using a 3-way split when you are also tuning hyperparameters.
- **Leave-One-Out Cross Validation (LOOCV)**: Splitting where the test set is just a single row (used for very small datasets).

## Code References
- `code/example-01-train-test-split.py` — Simple random train-test splitting.
- `code/example-02-kfold-cv.py` — K-Fold Cross Validation.
- `code/example-03-stratified-cv.py` — Stratified splitting for imbalanced data.


---

## Method & Options: Data Splitting

This document details the common scikit-learn methods used to split and validate datasets.

### 1. `sklearn.model_selection.train_test_split`

#### Purpose
The standard, go-to function for making a single random split of your data into training and testing sets.

#### Syntax
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### Common Arguments
- `arrays` (X, y): The datasets you want to split.
- `test_size` (float): The proportion of the dataset to include in the test split (e.g., `0.2` or `0.3`).
- `train_size` (float): Rarely used if `test_size` is defined.
- `random_state` (int): A seed value. Setting this to a number (like `42`) ensures that you get the exact same split every time you run the code. Crucial for reproducibility.
- `shuffle` (boolean): Default `True`. Whether to shuffle data before splitting.
- `stratify` (array-like): If not None, data is split in a stratified fashion, using this as the class labels. You usually pass `y` here.

#### Return Type
Returns a list containing the train-test split of the arrays you passed in (usually unpacked into 4 variables).

---

### 2. `sklearn.model_selection.KFold`

#### Purpose
Used to perform Cross-Validation. It divides the dataset into `k` consecutive folds. Each fold is used once as a validation while the `k - 1` remaining folds form the training set.

#### Syntax
```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X):
    # Training loop here
```

#### Common Arguments
- `n_splits` (int): Number of folds. Must be at least 2. Default is 5.
- `shuffle` (boolean): Whether to shuffle the data before splitting into batches.
- `random_state` (int): Used when `shuffle` is True.

#### Workflow
Usually, you don't write the `for` loop yourself. You pass the `KFold` object into a helper function like `cross_val_score`.

---

### 3. `sklearn.model_selection.cross_val_score`

#### Purpose
Evaluates a model's performance using cross-validation in one line of code, without writing manual loops.

#### Syntax
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator, X, y, cv=5, scoring='accuracy')
```

#### Common Arguments
- `estimator`: The machine learning model you want to evaluate (e.g., `LogisticRegression()`).
- `X`: The feature data.
- `y`: The target labels.
- `cv`: Determines the cross-validation splitting strategy. If an integer is passed, it uses `KFold` or `StratifiedKFold`.
- `scoring` (string): The metric to evaluate the model (e.g., `'accuracy'`, `'neg_mean_squared_error'`, `'r2'`).

#### Return Type
Returns an array of scores of the estimator for each run of the cross validation.

#### Best Practices
Always check the average (`scores.mean()`) and the variance (`scores.std()`) of the output array to see how stable your model is across different data splits.

---

## Data Splitting Examples

This document explains the python examples provided in the `code/` directory.

### 1. Simple Train-Test Split (`example-01-train-test-split.py`)
This script demonstrates the most basic form of model evaluation.
- It generates a dummy dataset (like predicting house prices).
- It splits the data 80% / 20%.
- It proves that the model performs perfectly on the data it memorized (train set), but slightly worse on data it has never seen (test set).

### 2. K-Fold Cross Validation (`example-02-kfold-cv.py`)
A single train-test split can be lucky or unlucky. This script demonstrates Cross-Validation.
- It splits the data into 5 chunks (folds).
- It trains the model 5 separate times, holding out a different chunk for testing each time.
- It prints the score for each fold, showing how model performance fluctuates depending on the data.
- It calculates the reliable "average" score.

### 3. Stratified Splitting (`example-03-stratified-cv.py`)
This is crucial for classification problems with imbalanced classes (e.g., 90% No Fraud, 10% Fraud).
- It generates a dataset with an intentional 9-to-1 imbalance.
- It shows how a normal random split might accidentally put 0% fraud cases in the test set.
- It uses `train_test_split(..., stratify=y)` to guarantee the 9-to-1 ratio exists perfectly in both the training set and the test set.

---

## Practice Exercises: Data Splitting

These exercises are designed to help you build muscle memory for splitting data correctly.

### Exercise 1: The Basic Split
1. Load the `iris` dataset from `sklearn.datasets`.
2. Assign the features to `X` and the target to `y`.
3. Use `train_test_split` to create a 75% train and 25% test split.
4. Set `random_state=42`.
5. Print the shape of `X_train` and `X_test` to verify the math (150 total rows * 0.25 = 37.5).

### Exercise 2: Understanding Random State
1. Create a simple list of numbers: `X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]` and `y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]`.
2. Run `train_test_split(X, y, test_size=0.2, random_state=1)` and print `y_train`.
3. Run it again with `random_state=1`. Did it change?
4. Run it again with `random_state=99`. What happened?
5. **Question**: Why is `random_state` so important when sharing code with a colleague?

### Exercise 3: The Stratification Problem
1. Look at the `y` array you created in Exercise 2. It has exactly 50% `0`s and 50% `1`s.
2. Run `train_test_split(X, y, test_size=0.4, random_state=4)`.
3. Look at `y_test`. Are there two `0`s and two `1`s? (Hint: probably not).
4. Now, run `train_test_split(X, y, test_size=0.4, random_state=4, stratify=y)`.
5. Look at `y_test`. What did `stratify` fix?

### Exercise 4: Cross-Validation Integration
1. Load the `diabetes` dataset (`load_diabetes()`) from sklearn.
2. Initialize a `LinearRegression` model.
3. Instead of splitting the data yourself, use `cross_val_score(model, X, y, cv=5)`.
4. Print the 5 scores.
5. Print the mean and standard deviation of the scores.

---

## Interview Questions: Data Splitting

### Beginner Questions
1. **What is the difference between a training set and a testing set?**
   - *Answer*: The training set is used to teach the model patterns in the data. The testing set is kept completely hidden from the model during training and is only used at the end to evaluate how well the model generalizes to new, unseen data.
2. **Why can't we just test the model on the same data we used to train it?**
   - *Answer*: Because the model might simply memorize the training data (overfitting). Testing on training data gives a falsely high accuracy score that will not hold up in production.

### Conceptual Questions
3. **What is Cross-Validation and why is it preferred over a simple train-test split?**
   - *Answer*: Cross-validation involves splitting the data into 'k' folds and training/testing the model 'k' times, rotating the test fold each time. It is preferred because a single train-test split can be biased by a "lucky" or "unlucky" random shuffle. CV gives a more reliable average performance metric.
4. **Explain what Data Leakage is in the context of data splitting.**
   - *Answer*: Data leakage occurs when information from outside the training dataset (i.e., from the test set) is used to create the model. A classic example is scaling/normalizing the *entire* dataset before splitting it, meaning the training data was influenced by the test data's mean/variance.
5. **What is stratification? When is it absolutely necessary?**
   - *Answer*: Stratification ensures that the train and test sets have the exact same proportion of class labels as the original dataset. It is absolutely necessary when dealing with highly imbalanced datasets (e.g., predicting a rare disease), ensuring the test set actually contains examples of the minority class.

### Practical / Scenario Questions
6. **You are building a time-series model to predict stock prices. How do you split your data?**
   - *Answer*: You **cannot** use a random `train_test_split`. If you randomly shuffle time-series data, you are letting the model "look into the future" to predict the past. You must use a chronological split (e.g., train on 2018-2022, test on 2023).
7. **You have a dataset of 10,000 patient X-rays. Some patients have 5 X-rays, some have 1. How do you split the data?**
   - *Answer*: You must use a "Group" split (like `GroupKFold`). You have to ensure that all X-rays from the *same* patient go entirely into either the train set or the test set. If Patient A's images are in both, the model might just learn to recognize Patient A's bone structure, not the actual disease.

---

## Python Code Examples

### `example-01-train-test-split.py`

```python
"""
Example 01: The Basic Train-Test Split
This script demonstrates how to split data and why it matters.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 1. Create a dummy dataset
# Let's say X is the square footage of a house, y is the price
np.random.seed(42)
X = np.random.rand(100, 1) * 2000 + 500  # Houses from 500 to 2500 sq ft
# Price is roughly $150 per sq ft, plus some random noise
y = X * 150 + (np.random.randn(100, 1) * 20000) 

# 2. Split the data
# We keep 20% of the data hidden for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Total records: {len(X)}")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print("-" * 30)

# 3. Train the model ONLY on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluate the model
# Let's see how well it memorized the training data
train_predictions = model.predict(X_train)
train_error = mean_absolute_error(y_train, train_predictions)

# Let's see how well it actually performs on unseen data
test_predictions = model.predict(X_test)
test_error = mean_absolute_error(y_test, test_predictions)

print(f"Error on Training Data (Memorized): ${train_error:,.2f}")
print(f"Error on Testing Data (Unseen):   ${test_error:,.2f}")

# Notice how the error is usually slightly higher on the testing data.
# This proves why we must evaluate models on unseen data!
```

### `example-02-kfold-cv.py`

```python
"""
Example 02: K-Fold Cross Validation
This script shows how to get a more reliable performance score by splitting data multiple times.
"""

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# 1. Load a standard dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# 2. Initialize the model
model = LinearRegression()

# 3. Perform 5-Fold Cross Validation
# Instead of splitting once, we split 5 times.
# scoring='r2' means we are checking the R-squared score (closer to 1.0 is better)
print("Running 5-Fold Cross Validation...")
scores = cross_val_score(model, X, y, cv=5, scoring='r2')

# 4. Analyze the results
print(f"Individual Scores for each fold: {np.round(scores, 3)}")

print("-" * 30)
print(f"Average (Reliable) Score: {scores.mean():.3f}")
print(f"Standard Deviation (Stability): {scores.std():.3f}")

# Note:
# Look at the individual scores. One of them might be 0.58, another might be 0.42.
# If we only did a single train_test_split, we might have accidentally gotten the "lucky" 0.58 split
# and thought our model was better than it really is.
# The Average Score (mean) gives us the honest truth.
```

### `example-03-stratified-cv.py`

```python
"""
Example 03: Stratified Splitting
This script demonstrates the solution for highly imbalanced datasets.
"""

import numpy as np
from sklearn.model_selection import train_test_split

# 1. Create a highly imbalanced dataset
# Imagine this is transaction data: 950 normal transactions, 50 fraudulent ones
X = np.random.rand(1000, 5) # 1000 rows, 5 features
y_normal = np.zeros(950)    # 0 = Normal
y_fraud = np.ones(50)       # 1 = Fraud
y = np.concatenate([y_normal, y_fraud])

print(f"Original Data Fraud Rate: {np.mean(y) * 100:.1f}%")
print("-" * 30)

# 2. A BAD Split (Random)
# We might accidentally put too few fraud cases in the test set.
X_train_bad, X_test_bad, y_train_bad, y_test_bad = train_test_split(X, y, test_size=0.2, random_state=12)

print("--- WITHOUT Stratification ---")
print(f"Train Set Fraud Rate: {np.mean(y_train_bad) * 100:.1f}%")
print(f"Test Set Fraud Rate:  {np.mean(y_test_bad) * 100:.1f}%")
print("Notice how the Test set doesn't represent the original 5% reality well!")
print()

# 3. A GOOD Split (Stratified)
# We force the split to maintain the exact same ratio of 0s and 1s
X_train_good, X_test_good, y_train_good, y_test_good = train_test_split(X, y, test_size=0.2, random_state=12, stratify=y)

print("--- WITH Stratification ---")
print(f"Train Set Fraud Rate: {np.mean(y_train_good) * 100:.1f}%")
print(f"Test Set Fraud Rate:  {np.mean(y_test_good) * 100:.1f}%")
print("Perfect! Both sets maintain the exact 5% ratio of the original data.")
```
