# Hyperparameter Tuning

## Learning Objective
Students will learn the difference between parameters and hyperparameters, and how to programmatically search for the optimal model configuration using Grid Search and Random Search to maximize performance without overfitting.

## What Is This Topic?
Every machine learning model has "settings" (like knobs and dials on a radio) that control how it learns. Hyperparameter tuning is the systematic process of twisting those knobs to find the combination that gives the best possible score on unseen data.

## Why This Topic Matters
The default settings in scikit-learn are usually "okay," but rarely optimal. A model with default settings might achieve 80% accuracy, but the exact same model with properly tuned hyperparameters could achieve 92%. Tuning is the step that takes a model from "functional" to "professional."

## Core Intuition
Imagine baking a cake.
- **Parameters**: The actual ingredients the cake absorbs as it bakes (the patterns the model learns from the data). You don't choose these; the oven (training process) figures it out.
- **Hyperparameters**: The temperature of the oven and the baking time. You *must* set these before you press start. If the temperature is too high, the cake burns (Overfitting). If too low, it's raw (Underfitting).

## Key Concepts

### 1. Parameters vs. Hyperparameters
- **Parameters**: Learned automatically during `model.fit()` (e.g., the slope $m$ and intercept $b$ in Linear Regression).
- **Hyperparameters**: Set manually by the Data Scientist *before* calling `model.fit()` (e.g., the `max_depth` of a Decision Tree, or the `learning_rate` of a Neural Network).

### 2. Grid Search (The Exhaustive Search)
You provide a list of values for each hyperparameter. The algorithm tries *every single possible combination*.
- **Pros**: Guaranteed to find the best combination out of the options you provided.
- **Cons**: Extremely slow. If you have 5 knobs with 5 settings each, that's $5^5 = 3125$ models to train!

### 3. Random Search (The Efficient Search)
You define a range for each hyperparameter. The algorithm randomly picks combinations for a set number of iterations (e.g., 100 times).
- **Pros**: Much faster. Surprisingly, it often finds a combination that is 99% as good as Grid Search in a fraction of the time.
- **Cons**: Not guaranteed to find the *absolute* mathematical best combination.

### 4. The Validation Set (Why we use CV)
You cannot tune your model using the Test Set. If you do, you are just repeatedly tweaking the model until it memorizes the Test Set! Therefore, tuning algorithms use **Cross-Validation** on the Training set to figure out the best settings.

## Output / Result Interpretation
The output of a tuning algorithm is simply the "best" model it found. You can then look at `best_params_` to see exactly which settings won the competition.

## Real-World Uses
- **Kaggle Competitions**: The difference between 1st place and 100th place is almost always who did a better job of hyperparameter tuning.
- **Deploying to Production**: Before deploying a final model to serve customers, it is standard practice to run an overnight Grid Search to squeeze out every last drop of performance.

## Common Mistakes
- **Tuning on the Test Set**: This completely ruins the integrity of your test set (Data Leakage).
- **Huge Grids**: Giving a Grid Search too many options and crashing your computer. Always start with a small Random Search to find the right "neighborhood", then do a tight Grid Search.
- **Ignoring the Defaults**: Sometimes, the default settings actually *are* the best. Always check if your tuned model actually beats the baseline model.

## Related Methods
- **Bayesian Optimization**: An advanced technique (using libraries like `Optuna` or `Hyperopt`) that learns from past attempts. If a certain combination was terrible, it avoids that "area" of settings on the next try.

## Code References
- `code/example-01-grid-search.py`
- `code/example-02-random-search.py`
- `code/example-03-nested-cv.py`
