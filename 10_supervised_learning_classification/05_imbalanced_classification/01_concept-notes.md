# Imbalanced Classification

## Learning Objective
Understand why standard classification models fail when one class significantly outnumbers another, why accuracy is a misleading metric, and how to use data-level and algorithmic-level techniques to solve this problem.

## What Is This Topic?
An "imbalanced dataset" occurs when the classes you are trying to predict are not represented equally. For example, in credit card fraud detection, 99.9% of transactions are legitimate (Class 0), and only 0.1% are fraudulent (Class 1). 

## Why This Topic Matters
Most real-world business problems are inherently imbalanced. Customers rarely click ads, machines rarely break down, and patients rarely have rare diseases. If you don't know how to handle imbalanced data, your models will be completely useless in the real world.

## The Core Problem: The "Accuracy Paradox"
If 99% of your data is Class 0, a "dumb" model that *always* predicts Class 0 will achieve 99% accuracy. 
To a beginner, 99% accuracy sounds amazing. To a business, the model is completely worthless because it completely failed to detect the 1% of cases that actually matter (the fraud).

## How to Fix It

There are generally three ways to handle imbalanced data:

### 1. Change the Metrics
Stop using Accuracy. Use metrics that focus on the minority class:
- **Precision**: Out of all the ones the model predicted as Fraud, how many were *actually* Fraud?
- **Recall (Sensitivity)**: Out of all the *actual* Fraud cases, how many did the model successfully find?
- **F1-Score**: The harmonic mean of Precision and Recall.

### 2. Algorithmic Changes (Class Weights)
Most Scikit-Learn models have a parameter called `class_weight`. By setting it to `'balanced'`, you tell the algorithm to pay *more attention* to the minority class. Mathematically, it applies a heavier penalty when the model gets a minority class prediction wrong.

### 3. Data-Level Changes (Resampling)
Change the data itself *before* feeding it to the model.
- **Undersampling**: Randomly delete rows from the majority class until the classes are equal. (Warning: You lose a lot of data).
- **Oversampling (SMOTE)**: Create fake, synthetic examples of the minority class to balance the dataset.

## What is SMOTE?
**S**ynthetic **M**inority **O**versampling **TE**chnique. 
Instead of just duplicating existing fraud rows, SMOTE looks at a fraud row, finds its nearest fraud neighbors, and generates a brand new, slightly different fraud row somewhere in between them.

## Golden Rule of Resampling
**NEVER resample your test set.** 
You only undersample or oversample the *training* data. The test data must remain a true, imbalanced representation of the real world, or else your evaluation metrics will be a lie.

## Code References
- `code/example-01-class-weights.py`
- `code/example-02-smote.py`
- `code/example-03-real-world-credit-card.py`
