# Interview Questions: Regression Metrics

## Beginner Questions
1. **Can you use Accuracy to evaluate a regression model? Why or why not?**
   - *Answer*: No. Accuracy requires exact matches. In regression, predicting $300,000 for a house when the real price is $300,001 is technically "wrong" but practically perfect. Regression metrics must measure the *distance* (error) of the prediction, not just right or wrong.
2. **What does an R-squared of 0.85 mean?**
   - *Answer*: It means that 85% of the variance in the target variable can be explained by the features in the model. The remaining 15% is due to unobserved variables or random noise.

## Conceptual Questions
3. **What is the difference between MAE and RMSE?**
   - *Answer*: MAE is the simple average of absolute errors. RMSE squares the errors before averaging them, and then takes the square root. Because it squares the errors, RMSE heavily penalizes large outliers.
4. **If your boss asks you to explain how far off your model's predictions are, which metric do you use: MSE, RMSE, or R-squared?**
   - *Answer*: You should use RMSE or MAE. MSE is in "squared units" which makes no sense to humans. R-squared is a percentage of variance, not a dollar amount. RMSE or MAE gives the error in the original units (e.g., "We are off by $10,000").
5. **Why do we need Adjusted R-squared?**
   - *Answer*: Standard R-squared will mechanically increase (or stay the same) every time you add a new feature to the model, even if the feature is completely random noise. Adjusted R-squared penalizes the score for adding useless features, giving a more honest assessment of the model's true predictive power.

## Practical / Scenario Questions
6. **You are building an AI to predict when a patient will wake up from anesthesia. If the AI predicts early, the doctors just wait a bit longer. If the AI predicts late, the patient might wake up during surgery (a catastrophe). Which metric do you optimize?**
   - *Answer*: You must heavily penalize large errors, especially in one direction. You would likely use RMSE (or a custom asymmetric loss function) because a single massive error is unacceptable. MAE is too forgiving of outliers.
7. **You calculated an R-squared score of -0.15. Is this a bug in your code?**
   - *Answer*: Not necessarily a bug. An R-squared of 0.0 means your model is exactly as good as a "dumb" model that just predicts the average of the dataset every single time. A negative R-squared means your model is actually *worse* than just predicting the average. The model has failed to learn any useful pattern.
