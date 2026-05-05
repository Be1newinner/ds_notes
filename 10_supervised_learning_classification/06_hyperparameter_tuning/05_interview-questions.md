# Interview Questions: Hyperparameter Tuning

## Beginner Questions
1. **What is the difference between a model parameter and a hyperparameter?**
   *Hint:* A model parameter is learned from the data during training (e.g., the weights in Logistic Regression or the splits in a Decision Tree). A hyperparameter is set by the data scientist *before* training begins (e.g., `max_depth` or `learning_rate`).
2. **What does Grid Search do?**
   *Hint:* It takes a list of possible hyperparameter values, creates every possible combination of them, trains a model for each combination, and returns the one that performs best.
3. **What does the "CV" in GridSearchCV stand for, and what does it mean?**
   *Hint:* Cross-Validation. It means the training data is split into multiple folds to evaluate the model safely without touching the final Test Set.

## Conceptual Questions
4. **Why is it considered bad practice to tune hyperparameters by evaluating the model on the final Test Set?**
   *Hint:* This causes Data Leakage. If you tweak the model until it gets a high score on the test set, the model is no longer generalizing to unseen data; it has essentially "memorized" the test set. The test set must be kept strictly separate until the very end.
5. **If you have a massive dataset and 10 hyperparameters to tune, why would you choose Randomized Search over Grid Search?**
   *Hint:* The "Curse of Dimensionality". Grid search time grows exponentially with the number of parameters. Randomized Search samples a fixed number of combinations, meaning it runs much faster and, statistically, often finds a combination very close to the optimal one.
6. **Explain what a Scikit-Learn `Pipeline` is and why it's useful for tuning.**
   *Hint:* A Pipeline bundles data preprocessing steps (like scaling) and the model into a single object. It is useful for tuning because it prevents data leakage during Cross-Validation.

## Practical Questions
7. **During a GridSearchCV with `cv=5`, does the scaler (e.g., StandardScaler) get fit on the entire training set before the folds are created, or is it fit 5 separate times?**
   *Hint:* If you do it correctly (using a Pipeline), the scaler is fit 5 separate times, using only the training folds in each step. If you scale the data *before* running GridSearchCV, you cause data leakage because information from the validation fold leaks into the scaling process.
8. **By default, what metric does `GridSearchCV` try to maximize for classification problems? How do you change it to focus on false negatives?**
   *Hint:* By default, it maximizes Accuracy. To focus on false negatives, you should set `scoring='recall'`.
9. **You run a Grid Search and the best `max_depth` found is 10. Your parameter grid was `[2, 4, 6, 8, 10]`. What should your next step be?**
   *Hint:* Because the best parameter was at the absolute edge of your grid, the true optimal value might be 12 or 15. You should run another search with a grid like `[10, 12, 14, 16]`.

## Output Interpretation
10. **You run a Grid Search. The model with `max_depth=5` scores 92% on the training folds and 91% on the validation folds. The model with `max_depth=15` scores 99% on the training folds and 91.5% on the validation folds. Which model should you choose and why?**
    *Hint:* You should choose `max_depth=5`. While the deeper tree has a tiny bit more validation accuracy, the massive gap between its training and validation score indicates it is severely overfitting. The simpler model is much more robust.
