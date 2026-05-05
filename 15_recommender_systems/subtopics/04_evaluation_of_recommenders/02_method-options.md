# Method Options and Properties: Evaluation of Recommenders

There is no single "magic function" in Scikit-Learn that calculates Recommender ranking metrics like Hit Ratio. They usually have to be calculated manually or using specialized libraries like `Surprise`.

## 1. Rating Prediction Metrics (Surprise Library)

### `surprise.accuracy.rmse(predictions)`
- **Purpose**: Calculates the Root Mean Squared Error.
- **Intuition**: Penalizes large errors heavily (e.g., predicting 5 stars when the reality was 1 star is punished much more than predicting 4 stars).

### `surprise.accuracy.mae(predictions)`
- **Purpose**: Calculates the Mean Absolute Error.
- **Intuition**: The average absolute difference between predicted and actual ratings. Easier to interpret than RMSE (e.g., "We are off by 0.7 stars on average").

## 2. Ranking Metrics (Manual or Information Retrieval Libraries)

### Precision at K (Precision@K)
- **Formula**: `(Recommended items that are relevant in Top K) / (K)`
- **Example**: If K=10, and 3 out of the 10 recommended movies were actually liked by the user, Precision@10 is 0.3.

### Recall at K (Recall@K)
- **Formula**: `(Recommended items that are relevant in Top K) / (Total relevant items for the user)`
- **Example**: If K=10, we recommended 3 good movies. But the user has a total of 15 favorite movies. Recall@10 is 3/15 = 0.2.

### Hit Ratio (HR)
- **Workflow for Leave-One-Out Evaluation**:
  1. For each user, extract one item they liked and put it in a "Test Set".
  2. Put 99 items they *never* interacted with into the Test Set.
  3. The model ranks these 100 items.
  4. If the 1 item they actually liked appears in the Top-K of that ranked list, it's a "Hit" (1). Otherwise, a "Miss" (0).
  5. `Hit Ratio = Sum of Hits / Total Users`

## 3. Scikit-Learn Ranking Metrics

### `sklearn.metrics.ndcg_score`
- **Purpose**: Calculates Normalized Discounted Cumulative Gain.
- **Syntax**: `ndcg_score(y_true, y_score, k=10)`
- **Intuition**: Measures the quality of the ranking. Relevant items at the very top of the list contribute much more to the score than relevant items at the bottom.
- **Common Usage**: Pass the actual relevance scores and the predicted probabilities/ratings, and set K to the length of your recommendation list.

## Common Mistakes
- **Confusing Classification Precision with Recommender Precision@K**: In classification, precision is `TP / (TP + FP)`. In recommenders, the denominator is explicitly restricted to `K` (the number of slots shown to the user).
- **Evaluating on the Train Set**: Always ensure that the items being evaluated in Top-K were completely hidden from the model during the training phase.
