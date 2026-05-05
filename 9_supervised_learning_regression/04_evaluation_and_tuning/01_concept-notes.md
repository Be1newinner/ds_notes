# Evaluation and Tuning for Regression Models

## Learning Objective
Students will learn how to quantify the errors made by regression models, translate those mathematical errors into business logic, and use hyperparameter tuning to optimize model performance.

## What Is This Topic?
Evaluation metrics tell us *how wrong* our model's predictions are. Tuning is the process of adjusting the "knobs and dials" (hyperparameters) of an algorithm to make those errors as small as possible without overfitting.

## Why This Topic Matters
An R-squared of 0.90 sounds great, but if you are predicting heart rates, an error of 10 beats per minute might be fatal. You must know how to translate statistical metrics into real-world impact. Tuning separates a decent model from a production-ready model.

## Core Evaluation Metrics
- **Mean Absolute Error (MAE)**: The average of the absolute differences between predictions and actual values. Highly interpretable (e.g., "We are off by $500 on average").
- **Mean Squared Error (MSE)**: The average of the *squared* differences. Heavily penalizes large errors. Not in the original unit of measurement (e.g., "Dollars squared").
- **Root Mean Squared Error (RMSE)**: The square root of MSE. Brings the error back to the original unit (e.g., "$"). Best default metric because it penalizes large errors while remaining interpretable.
- **R-squared ($R^2$)**: The proportion of variance in the target that can be explained by the features. 1.0 is perfect, 0.0 is predicting the mean, negative means worse than predicting the mean.
- **Adjusted R-squared**: Adjusts $R^2$ based on the number of features. Prevents you from artificially inflating $R^2$ by just throwing useless features at the model.

## Core Tuning Concepts
- **Hyperparameters**: Settings you configure *before* training (like `max_depth` in trees or `alpha` in Ridge). The model does *not* learn these.
- **Grid Search**: Trying every single combination of hyperparameters in a defined grid. Exhaustive but very slow.
- **Randomized Search**: Trying a random sample of combinations. Much faster and often finds a "good enough" solution.
- **Cross-Validation (CV)**: Splitting the training data into $K$ folds. Train on $K-1$, validate on the remaining 1. Repeat $K$ times. Ensures your tuning results aren't just lucky on one specific train/test split.

## Real-World Uses
- Choosing between a fast Linear model and a slow Random Forest by comparing their RMSE and prediction latency.
- Spending compute time tuning an XGBoost model to squeeze out an extra 1% accuracy in a high-frequency trading algorithm where 1% equals millions of dollars.

## Common Mistakes
- Relying entirely on $R^2$. Always look at MAE or RMSE to understand the physical magnitude of the error.
- Tuning hyperparameters on the Test Set. This causes "data leakage." You must tune using Cross-Validation on the Training Set, and only evaluate on the Test Set at the very end.
- Searching too broad a grid with Grid Search, causing the code to run for days.

## Code References
- `code/example-01-metrics.py`
- `code/example-02-gridsearch.py`
