# Interview Questions: Hyperparameter Tuning

## Beginner Questions
1. **What is the difference between a Parameter and a Hyperparameter?**
   - *Answer*: A parameter is learned automatically by the model from the data (like the weights in a regression equation). A hyperparameter is set manually by the data scientist *before* training begins (like the maximum depth of a tree).
2. **What does a Grid Search actually do?**
   - *Answer*: It takes a dictionary of different hyperparameter values, generates every single possible combination of those values, and trains a model for each combination to see which one performs the best.

## Conceptual Questions
3. **Why do we need a Validation set (or Cross-Validation) during hyperparameter tuning? Why not just use the Test set?**
   - *Answer*: If you tune your hyperparameters to get the highest score on the Test set, you are causing "Data Leakage." You are essentially manually tweaking the model until it memorizes the Test data. The Test set must remain completely hidden until the very end. Therefore, we use a separate Validation set (or CV on the training data) to pick the best hyperparameters.
4. **When would you choose Random Search over Grid Search?**
   - *Answer*: When the hyperparameter space is very large or continuous. Grid Search on 5 hyperparameters with 10 options each requires 100,000 model trainings. Random Search can explore the same space and often find a near-optimal solution in just 100 or 200 iterations, saving immense amounts of time.
5. **If I run Grid Search and the winning hyperparameters are at the absolute edge of my grid (e.g., I searched `max_depth` [2, 4, 6] and 6 won), what should I do?**
   - *Answer*: You should expand your grid in that direction! If 6 won, the true optimal value might be 8 or 10. You should run another search checking [6, 8, 10, 12].

## Practical / Scenario Questions
6. **You run a Grid Search with `cv=5` on a grid with 100 combinations. How many times is the `fit()` method called in total?**
   - *Answer*: 500 times. (100 combinations * 5 folds). This is why tuning can be computationally expensive.
7. **Your baseline model got 85% accuracy. You spent 3 days running a massive Grid Search, and the tuned model gets 85.1% accuracy. Should you deploy the tuned model?**
   - *Answer*: Usually, no. The principle of Occam's Razor (or the "Simplest Solution") applies. If tuning only provides a microscopic benefit, it is usually better to deploy the simpler, baseline model, as the heavily tuned model might be slightly overfitting to the training data.
