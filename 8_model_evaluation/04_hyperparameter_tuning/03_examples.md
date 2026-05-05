# Hyperparameter Tuning Examples

This document explains the python examples provided in the `code/` directory.

## 1. Grid Search (`example-01-grid-search.py`)
This script demonstrates the "brute force" method of finding the best model.
- It loads a dataset and initializes a Random Forest model.
- It defines a `param_grid` with a few different settings for the forest.
- It uses `GridSearchCV` to try every single combination, using cross-validation to ensure the results are robust.
- It extracts the `best_estimator_` and shows how much better it performs compared to a default model.

## 2. Random Search (`example-02-random-search.py`)
This script demonstrates the "smart and fast" method.
- It uses `scipy.stats` to define *ranges* of numbers (e.g., any number between 10 and 1000) rather than a hardcoded list.
- It uses `RandomizedSearchCV` to try exactly 20 random combinations.
- It demonstrates that Random Search usually finds a near-perfect model much faster than an exhaustive Grid Search would take on the same ranges.

## 3. Nested Cross-Validation (`example-03-nested-cv.py`)
This is an advanced concept for when you want to report the absolute most unbiased performance estimate possible.
- Standard Grid Search uses CV to *tune* the model, but it still evaluates the final model on a single static test set.
- Nested CV puts a Grid Search *inside* another Cross-Validation loop.
- It prevents "overfitting to the validation set" and provides a highly conservative estimate of how the model will perform in the wild.
