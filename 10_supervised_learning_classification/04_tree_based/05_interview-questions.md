# Interview Questions: Tree-Based Methods

## Beginner Questions
1. **Explain how a Decision Tree works to a non-technical person.**
   *Hint:* It's like playing a game of 20 questions. It asks a series of True/False questions about the data to narrow down the possibilities until it makes a final guess.
2. **What is a Random Forest?**
   *Hint:* It is an ensemble (collection) of many different decision trees. They all make a prediction, and the forest takes a majority vote to decide the final answer.
3. **Do you need to scale or normalize data before using a Random Forest?**
   *Hint:* No. Trees only care about order and splitting points (e.g., "Is Salary > 50,000?"). The absolute scale of the number doesn't matter at all.

## Conceptual Questions
4. **Why is a single Decision Tree almost never used in production?**
   *Hint:* It is highly prone to overfitting. If allowed to grow deep enough, it will perfectly memorize the training data and fail miserably on new, unseen data.
5. **How does a Random Forest ensure that its trees are actually different from one another?**
   *Hint:* Through two types of randomness: 
   1) **Row sampling (Bagging):** Each tree gets a random sample of the training rows. 
   2) **Feature sampling:** At every single split, the tree is only allowed to choose from a random subset of the features (usually the square root of the total number of features).
6. **What is the fundamental difference in how Random Forests and Gradient Boosting Machines (GBMs) build their trees?**
   *Hint:* Random Forests build trees *independently and in parallel* (Bagging). GBMs build trees *sequentially*, where each new tree tries to correct the errors made by the previous trees (Boosting).

## Practical Questions
7. **If your Random Forest is overfitting, what hyperparameters should you tune?**
   *Hint:* Decrease `max_depth` to make the trees shallower. Increase `min_samples_split` or `min_samples_leaf` to prevent nodes from splitting on very few data points. 
8. **What does the `learning_rate` parameter do in a Gradient Boosting model?**
   *Hint:* It scales the contribution of each newly added tree. A lower learning rate means each tree contributes less to the final prediction, making the model more robust and less prone to overfitting, but it will require more trees (`n_estimators`) to train.
9. **How do tree-based models calculate Feature Importance?**
   *Hint:* They look at every node where a specific feature was used to make a split, and calculate how much that split decreased the "impurity" (Gini or Entropy). The more a feature decreases impurity across all trees in the forest, the more important it is.

## Output Interpretation
10. **Your Random Forest model has a feature importance score of 0.00 for the feature "Customer Age". What does this mean?**
    *Hint:* It means that across all 100+ trees in the forest, "Customer Age" was never used to make a split, or if it was, it didn't improve the purity of the classification at all. You could likely drop this feature without hurting model performance.
