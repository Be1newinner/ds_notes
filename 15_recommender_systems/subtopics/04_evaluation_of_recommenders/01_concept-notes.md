# Evaluation of Recommenders

## Learning Objective
What students should understand after this lesson:
- Why standard classification/regression metrics (Accuracy, MSE) are often insufficient for recommenders.
- The difference between Prediction metrics (RMSE, MAE) and Ranking metrics (Precision@k, Recall@k, NDCG).
- What a "Hit Ratio" is.
- How to evaluate offline vs online (A/B testing).

## What Is This Topic?
Evaluating a recommendation system is fundamentally different from evaluating a standard machine learning model. In standard ML, we care about predicting every single point accurately. In Recommenders, users only see the Top 10 items. We don't care if the system accurately predicts that a user will hate a movie (1 star); we only care if the 10 movies we *showed* them were actually 5-star movies.

## Why This Topic Matters
If you optimize for the wrong metric, you build a bad product. If you optimize purely for RMSE (Root Mean Squared Error), your model might learn to safely predict 3.5 stars for everything. This minimizes error but results in a boring, useless recommendation feed. You must measure what the user actually sees.

## Core Intuition
Imagine you have 10,000 movies.
- **Prediction**: Trying to guess the exact rating out of 5 for all 10,000 movies.
- **Ranking**: Sorting the 10,000 movies from best to worst and seeing if the top 10 movies in your list are actually movies the user loves.

## Key Concepts
- **RMSE / MAE**: Rating prediction error. How far off was our 1-5 star guess?
- **Top-K Recommendations**: The list of K items (usually 5, 10, or 20) actually shown to the user.
- **Hit Ratio**: Out of all users, what percentage were recommended an item in their Top-K that they actually ended up clicking/buying?
- **Precision@K**: Out of the Top K items we recommended, what percentage were actually relevant/liked?
- **Recall@K**: Out of all the items the user actually liked, what percentage did we manage to put in the Top K?
- **NDCG (Normalized Discounted Cumulative Gain)**: A metric that cares about *order*. It gives a higher score if the relevant item is at position #1 rather than position #10.

## Step-by-Step Explanation
1. **Hold-out Set**: Hide some of the user's known interactions (e.g., hide the last movie they watched). This is the Test Set.
2. **Train**: Train the model on the remaining data.
3. **Generate Top-K**: Ask the model to generate a Top 10 list for the user.
4. **Evaluate**: 
   - Did the hidden movie appear in the Top 10? (If yes, it's a Hit).
   - How high up the list was it? (Use NDCG).

## Important Parameters / Options / Settings
- **K**: The length of the recommendation list. Usually matches the UI of the application (e.g., Netflix shows 10 items in a row, so K=10).
- **Relevance Threshold**: For explicit ratings (1-5), you must define what "relevant" means. Usually, >= 4 stars is relevant.

## Output / Result Interpretation
- **Hit Ratio**: A Hit Ratio of 0.10 means 10% of users interacted with at least one item from their Top-K list.
- **RMSE**: An RMSE of 0.85 means our star rating prediction is off by 0.85 stars on average.

## Real-World Uses
- **Offline Evaluation**: Using historical data to compute Hit Ratio and NDCG before deploying.
- **Online Evaluation**: A/B testing two algorithms in production and measuring actual Click-Through Rate (CTR) and Revenue.

## Advantages of Ranking Metrics (Precision@K)
- Closely mimics the actual user experience. Users don't care about the 1,000th item on the list.

## Limitations of Offline Metrics
- **Historical Bias**: You can only evaluate based on what the user *did* in the past. If you recommend a brilliant new item they've never seen, offline metrics will count it as a "Miss" (error), even if they would have loved it in real life. This is why Online A/B testing is mandatory.

## Common Mistakes
- **Relying entirely on RMSE**: The famous Netflix Prize was based on RMSE, but the winning algorithm was never fully deployed because it was too complex and predicting exact ratings didn't translate to a better Top-10 ranking experience.
- **Using Accuracy**: Recommender utility matrices are 99% zeros. If you predict 0 (no interaction) for everything, you will get 99% accuracy. Accuracy is a useless metric here.

## Related Methods
- A/B Testing (Business Statistics module)
- Precision/Recall (Classification evaluation module)

## Code References
- `code/example-01-basic.py` — Calculating RMSE and MAE.
- `code/example-02-intermediate.py` — Calculating Hit Ratio and Precision@K.
