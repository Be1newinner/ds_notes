# Hyperparameter Tuning: Search Strategies & Workflows

## Learning Objective
Learn how to systematically find the best mathematical settings (hyperparameters) for a machine learning model to maximize its performance without overfitting, using techniques like Grid Search and Random Search.

## What Is This Topic?
When you initialize a model (e.g., `RandomForestClassifier()`), it uses default settings provided by Scikit-Learn. These settings are called **Hyperparameters** (like `max_depth`, `n_estimators`, `learning_rate`). They are settings you configure *before* training. Hyperparameter Tuning is the automated process of trying many different combinations of these settings to find the one that works best for your specific dataset.

*(Note: Parameters, like the weights in Logistic Regression, are learned by the algorithm during training. Hyperparameters are set by the data scientist).*

## Why This Topic Matters
The default settings almost never yield the best possible model. Tuning hyperparameters is how you squeeze the last 5% to 15% of performance out of an algorithm. It is a mandatory step in any professional ML pipeline.

## The Problem with Manual Tuning
You could manually change `max_depth` to 5, train the model, check the score, then change it to 6, train, check, etc. But if you have 4 different hyperparameters to tune, the number of possible combinations explodes. You need an automated strategy.

## Strategy 1: Grid Search
Grid Search is the brute-force approach. You give it a list of values for each hyperparameter (a "grid"). It will train a model for **every single possible combination** and tell you which one was best.
- **Pros**: Guaranteed to find the absolute best combination within the grid you provided.
- **Cons**: Computationally massive. If you have 5 parameters with 5 values each, that's $5^5 = 3,125$ models to train!

## Strategy 2: Randomized Search
Instead of trying every combination, you give it ranges of values, and it randomly selects combinations a set number of times (e.g., "try 100 random combinations").
- **Pros**: Much faster than Grid Search. Statistically, it is highly likely to find a combination that is very close to the optimal one, in a fraction of the time.
- **Cons**: Not guaranteed to find the *absolute* best combination.

## The Role of Cross-Validation (The "CV" in GridSearchCV)
If you tune your model by testing it on your Test Set, you are cheating. You are optimizing the model to specifically pass the test. 
To avoid this, we use **K-Fold Cross-Validation**:
1. Take the Training Set and split it into $K$ chunks (folds), say 5.
2. Train the model on 4 folds, evaluate on the 1 remaining fold.
3. Repeat this 5 times, so every fold gets to be the evaluation set once.
4. Average the 5 scores.
This allows us to evaluate model configurations thoroughly without ever touching the true Test Set.

## Advanced Workflows
In modern Data Science, tuning doesn't just happen on the model. You tune the entire pipeline. For example, you can use Grid Search to find out if `StandardScaler` works better than `MinMaxScaler`, while simultaneously finding the best `C` value for an SVM.

## Code References
- `code/example-01-grid-search.py`
- `code/example-02-random-search.py`
- `code/example-03-advanced-workflow.py`
