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


---

## Method Options and Properties: Evaluation of Recommenders

There is no single "magic function" in Scikit-Learn that calculates Recommender ranking metrics like Hit Ratio. They usually have to be calculated manually or using specialized libraries like `Surprise`.

### 1. Rating Prediction Metrics (Surprise Library)

#### `surprise.accuracy.rmse(predictions)`
- **Purpose**: Calculates the Root Mean Squared Error.
- **Intuition**: Penalizes large errors heavily (e.g., predicting 5 stars when the reality was 1 star is punished much more than predicting 4 stars).

#### `surprise.accuracy.mae(predictions)`
- **Purpose**: Calculates the Mean Absolute Error.
- **Intuition**: The average absolute difference between predicted and actual ratings. Easier to interpret than RMSE (e.g., "We are off by 0.7 stars on average").

### 2. Ranking Metrics (Manual or Information Retrieval Libraries)

#### Precision at K (Precision@K)
- **Formula**: `(Recommended items that are relevant in Top K) / (K)`
- **Example**: If K=10, and 3 out of the 10 recommended movies were actually liked by the user, Precision@10 is 0.3.

#### Recall at K (Recall@K)
- **Formula**: `(Recommended items that are relevant in Top K) / (Total relevant items for the user)`
- **Example**: If K=10, we recommended 3 good movies. But the user has a total of 15 favorite movies. Recall@10 is 3/15 = 0.2.

#### Hit Ratio (HR)
- **Workflow for Leave-One-Out Evaluation**:
  1. For each user, extract one item they liked and put it in a "Test Set".
  2. Put 99 items they *never* interacted with into the Test Set.
  3. The model ranks these 100 items.
  4. If the 1 item they actually liked appears in the Top-K of that ranked list, it's a "Hit" (1). Otherwise, a "Miss" (0).
  5. `Hit Ratio = Sum of Hits / Total Users`

### 3. Scikit-Learn Ranking Metrics

#### `sklearn.metrics.ndcg_score`
- **Purpose**: Calculates Normalized Discounted Cumulative Gain.
- **Syntax**: `ndcg_score(y_true, y_score, k=10)`
- **Intuition**: Measures the quality of the ranking. Relevant items at the very top of the list contribute much more to the score than relevant items at the bottom.
- **Common Usage**: Pass the actual relevance scores and the predicted probabilities/ratings, and set K to the length of your recommendation list.

### Common Mistakes
- **Confusing Classification Precision with Recommender Precision@K**: In classification, precision is `TP / (TP + FP)`. In recommenders, the denominator is explicitly restricted to `K` (the number of slots shown to the user).
- **Evaluating on the Train Set**: Always ensure that the items being evaluated in Top-K were completely hidden from the model during the training phase.

---

## Code Examples: Evaluation of Recommenders

These examples demonstrate how to evaluate recommender models using both standard ML prediction error metrics and specialized ranking metrics.

### Code References

- **`code/example-01-basic.py`** — Calculating standard prediction error metrics (RMSE and MAE) manually.
- **`code/example-02-intermediate.py`** — Implementing custom functions to calculate Hit Ratio, Precision@K, and Recall@K, which are the true measures of a Recommender's success.

---

## Practice: Evaluation of Recommenders

### Exercise 1: RMSE Calculation
You have the following true ratings and predicted ratings:
- Item 1: True=5, Pred=4
- Item 2: True=2, Pred=2
- Item 3: True=1, Pred=3

1. Calculate the MAE (Mean Absolute Error).
2. Calculate the MSE (Mean Squared Error).
3. Calculate the RMSE (Root Mean Squared Error).

### Exercise 2: Precision@K and Recall@K
A user has 4 movies in their "Favorites" list: [A, B, C, D].
Your model generates a Top 5 recommendation list for this user: [X, B, Y, Z, D].

1. Calculate Precision@5.
2. Calculate Recall@5.

### Exercise 3: The Hit Ratio Concept
You are doing a "Leave-One-Out" evaluation. You hid a movie the user watched (`Movie_Target`). You mixed `Movie_Target` with 99 random movies the user has never watched. Your model ranks these 100 movies.
`Movie_Target` appears at rank #12.

1. If you are calculating Hit Ratio for K=10 (Top 10), is this a Hit (1) or a Miss (0)?
2. If you change your UI to show a Top 20 list (K=20), is it a Hit or a Miss?

### Exercise 4: Python Coding Challenge
Write a Python function `precision_at_k(recommended_list, actual_list, k)` that takes a list of recommended item IDs, a list of actual liked item IDs, and an integer K, and returns the Precision@K metric.

---

## Interview Questions: Evaluation of Recommenders

### Beginner Questions
1. **Why is Accuracy not a good metric for a Recommender System?**
   *Answer*: Because the User-Item matrix is 99% zeros (unrated items). If a model predicts that the user won't interact with any item, it achieves 99% accuracy, but it recommends absolutely nothing.
2. **What does "RMSE" stand for and what does it measure in this context?**
   *Answer*: Root Mean Squared Error. It measures the average difference between the predicted rating (e.g., 4.5 stars) and the actual rating (e.g., 3.0 stars) across all predictions.

### Conceptual Questions
3. **What is the difference between Prediction Metrics and Ranking Metrics?**
   *Answer*: Prediction metrics (RMSE, MAE) evaluate how closely the model guesses the exact rating. Ranking metrics (Precision@K, Hit Ratio, NDCG) evaluate whether the items the user *actually* wants to see are placed at the very top of the recommendation list.
4. **Explain what NDCG is trying to measure.**
   *Answer*: Normalized Discounted Cumulative Gain measures the quality of a ranking. It assumes that an item placed at rank #1 is much more valuable to the user than the exact same item placed at rank #10, and discounts the score logarithmically the further down the list you go.

### Practical Questions
5. **You improve your model's RMSE from 0.85 to 0.70 offline, but when you deploy it, Click-Through Rate drops by 5%. Why might this happen?**
   *Answer*: Optimizing for RMSE often causes the model to recommend very "safe", universally well-rated items (like classic movies) to everyone. While these predictions are mathematically accurate, they are boring to users who want personalized, novel recommendations, leading to fewer clicks.
6. **How would you perform a Leave-One-Out evaluation to calculate Hit Ratio@10?**
   *Answer*: For each user, I would take one item they interacted with and set it aside as the target. I would then take 99 items they never interacted with. I would ask the model to score all 100 items and sort them. If the true target item appears in the Top 10 ranks, I record a Hit (1). Otherwise, a Miss (0). The average across all users is the Hit Ratio@10.

---

## Python Code Examples

### `example-01-basic.py`

```python
import numpy as np

# Example 01: Rating Prediction Metrics
# Calculating RMSE and MAE manually to understand how they work.

print("--- Rating Prediction Metrics ---")

# 1. Mock Data: True Ratings vs Model Predicted Ratings
# Imagine a user rated 5 movies.
true_ratings = np.array([5.0, 4.0, 2.0, 1.0, 3.0])
predictions = np.array([4.0, 4.5, 3.0, 1.0, 4.5])

print(f"True Ratings:      {true_ratings}")
print(f"Predicted Ratings: {predictions}\n")

# 2. Calculate Mean Absolute Error (MAE)
# MAE is the simple average of the absolute errors.
absolute_errors = np.abs(true_ratings - predictions)
mae = np.mean(absolute_errors)

print("Absolute Errors for each movie:")
print(absolute_errors)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print("Meaning: On average, our predictions are off by 0.7 stars.\n")

# 3. Calculate Root Mean Squared Error (RMSE)
# RMSE squares the errors before averaging them, which punishes large mistakes heavily.
squared_errors = np.square(true_ratings - predictions)
mse = np.mean(squared_errors)
rmse = np.sqrt(mse)

print("Squared Errors for each movie:")
print(squared_errors)
# Notice how the last movie (off by 1.5 stars) generated a squared error of 2.25!
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Conclusion: RMSE will always be larger than or equal to MAE. 
# Recommender competitions (like Netflix) usually use RMSE to heavily penalize wild guesses.
```

### `example-02-intermediate.py`

```python
# Example 02: Ranking Metrics (Precision@K and Hit Ratio)
# How to evaluate what the user actually sees.

print("--- Ranking Metrics ---")

# 1. Precision@K and Recall@K
def precision_recall_at_k(recommended_items, actual_liked_items, k=5):
    # Slice the recommendation list to only look at the Top K items
    top_k_recs = recommended_items[:k]
    
    # Find the intersection (items recommended AND actually liked)
    hits = set(top_k_recs).intersection(set(actual_liked_items))
    
    # Precision@K: Out of K recommendations, how many were hits?
    precision = len(hits) / k
    
    # Recall@K: Out of all possible liked items, how many did we find in Top K?
    if len(actual_liked_items) == 0:
        recall = 0.0
    else:
        recall = len(hits) / len(actual_liked_items)
        
    return precision, recall

# Let's test it
# The system recommended 10 items in this exact order
recs = ['ItemA', 'ItemB', 'ItemC', 'ItemD', 'ItemE', 'ItemF', 'ItemG', 'ItemH', 'ItemI', 'ItemJ']

# The user actually only liked 3 items
actual_likes = ['ItemC', 'ItemZ', 'ItemE']

# Let's see how our Top 5 list performed
p_at_5, r_at_5 = precision_recall_at_k(recs, actual_likes, k=5)

print("Top 5 Recommendations: ", recs[:5])
print("Actual Liked Items:    ", actual_likes)
print(f"\nPrecision@5: {p_at_5:.2f} (2 out of 5 recs were good)")
print(f"Recall@5:    {r_at_5:.2f} (We found 2 out of 3 total good items)")


# 2. Hit Ratio (Leave-One-Out Evaluation)
print("\n--- Hit Ratio (Leave-One-Out) ---")
# Imagine for User 1, we hid "ItemZ". We mix it with 9 random items and rank them.
# The ranked list generated by the model is below:
ranked_list = ['Item1', 'Item2', 'ItemZ', 'Item4', 'Item5', 'Item6', 'Item7', 'Item8', 'Item9', 'Item10']
target_item = 'ItemZ'

def check_hit(ranked_list, target_item, k=3):
    top_k = ranked_list[:k]
    if target_item in top_k:
        return 1
    return 0

print(f"Target Item: {target_item}")
print(f"Model Ranked List: {ranked_list}")

hit_at_1 = check_hit(ranked_list, target_item, k=1)
hit_at_3 = check_hit(ranked_list, target_item, k=3)
hit_at_5 = check_hit(ranked_list, target_item, k=5)

print(f"\nHit Ratio@1: {hit_at_1} (Target was not #1)")
print(f"Hit Ratio@3: {hit_at_3} (Target was #3, so it's a Hit!)")
print(f"Hit Ratio@5: {hit_at_5}")

# To get the final Hit Ratio for your model, you run this for every user and average the results.
```
