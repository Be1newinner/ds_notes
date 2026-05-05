# Interview Questions: Evaluation and Tuning

## Beginner Level
1. **What is the difference between MAE and RMSE?**
   - *Expected Answer*: MAE is the simple average of the absolute errors. RMSE squares the errors before averaging them, and then takes the square root. Because it squares the errors, RMSE heavily penalizes large errors compared to MAE.

2. **What does an R-squared of 0 mean? What about a negative R-squared?**
   - *Expected Answer*: An R-squared of 0 means the model is no better than just predicting the mean (average) of the target variable for every single observation. A negative R-squared means the model is actually performing *worse* than a flat line predicting the mean.

## Intermediate Level
3. **What is Adjusted R-squared and why is it preferred over R-squared in Multiple Linear Regression?**
   - *Expected Answer*: Normal R-squared will either stay the same or increase every time you add a new feature, even if the feature is total garbage. Adjusted R-squared penalizes the model for adding features that don't genuinely improve predictions, preventing you from artificially inflating the score.

4. **Explain how K-Fold Cross-Validation works.**
   - *Expected Answer*: The training data is split into K equal parts (folds). The model is trained on K-1 folds and validated on the 1 remaining fold. This process is repeated K times, with each fold serving as the validation set exactly once. The final validation score is the average of the K scores.

## Advanced / Practical Level
5. **If you have a massive dataset (10 million rows) and 15 hyperparameters to tune, would you use GridSearchCV? Why or why not?**
   - *Expected Answer*: Absolutely not. Grid search evaluates every single possible combination. With that much data and that many parameters, it would take weeks or months to compute. I would use RandomizedSearchCV to test a random subset of combinations, or Bayesian Optimization (like Optuna) to intelligently search the space.

6. **What is "Data Leakage" during tuning?**
   - *Expected Answer*: It occurs when you tune hyperparameters using the Test Set. If you try 100 combinations and pick the one that scores highest on the Test Set, your model hasn't truly generalized—you've just manually selected the parameters that memorize that specific Test Set. The Test Set must remain completely unseen until the final evaluation.
