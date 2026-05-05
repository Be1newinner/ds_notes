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
