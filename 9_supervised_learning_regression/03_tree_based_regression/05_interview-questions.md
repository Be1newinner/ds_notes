# Interview Questions: Tree-Based Regression

## Beginner Level
1. **How does a Decision Tree make a prediction for a continuous target variable?**
   - *Expected Answer*: It traverses the tree by answering the feature splits (yes/no questions) until it reaches a leaf node. The prediction is the average (mean) of all the training target values that fell into that specific leaf node.

2. **What is a Random Forest?**
   - *Expected Answer*: It is an ensemble method that builds many decision trees and averages their predictions. To ensure the trees are diverse, it trains each tree on a random sample of the data (bootstrapping) and looks at a random subset of features at each split.

## Intermediate Level
3. **What is the difference between Bagging and Boosting?**
   - *Expected Answer*: Bagging (like Random Forest) builds multiple independent models in parallel and averages them to reduce variance. Boosting (like XGBoost) builds models sequentially, where each new model tries to correct the errors (residuals) of the combined previous models, reducing both bias and variance.

4. **Why are Tree-based models robust to outliers?**
   - *Expected Answer*: Because they split data based on thresholds (e.g., "is income > $100k?"). Whether an income is $105k or $10 million, it falls into the same bucket and doesn't pull a "line of best fit" toward it like it would in linear regression.

5. **Can a Random Forest Regressor extrapolate? (Predict a value higher than any value in its training set?)**
   - *Expected Answer*: No. Because the prediction at a leaf node is the average of the training samples in that node, it is mathematically impossible for a tree to predict a value higher than the maximum (or lower than the minimum) value seen in the training data.

## Advanced / Practical Level
6. **If your Gradient Boosting model (XGBoost) is overfitting, what parameters would you tune?**
   - *Expected Answer*: I would lower the `learning_rate` (and proportionally increase `n_estimators`), decrease `max_depth` to make individual trees shallower, or use subsampling parameters like `subsample` or `colsample_bytree` to introduce more randomness.

7. **How is Feature Importance calculated in a Random Forest?**
   - *Expected Answer*: It is usually calculated based on the "Gini importance" or "Mean Decrease in Impurity." Every time a feature is used to split a node, it calculates how much that split reduced the variance (or impurity) compared to the parent node. The total variance reduction across all trees for that feature is averaged to get the final importance score.
