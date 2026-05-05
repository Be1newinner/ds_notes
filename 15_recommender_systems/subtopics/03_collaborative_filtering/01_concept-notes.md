# Collaborative Filtering

## Learning Objective
What students should understand after this lesson:
- The intuition behind "wisdom of the crowd" in recommendations.
- The difference between User-User and Item-Item Collaborative Filtering (Memory-Based).
- The basics of Matrix Factorization / SVD (Model-Based).
- How to handle the Cold Start problem for CF.

## What Is This Topic?
Collaborative Filtering (CF) recommends items based on the behavior of *other users*. It assumes that if User A and User B agreed on past items, they will agree on future items. It does not look at the content of the items at all; it only looks at user-item interactions (ratings, clicks, purchases).

## Why This Topic Matters
CF is the engine behind some of the most successful recommendation systems in history, like Amazon's "Customers who bought this also bought..." feature. It's powerful because it can discover hidden patterns and recommend things a user never would have searched for explicitly.

## Core Intuition
- **User-Based CF**: Find a "neighbor" who has almost identical taste to you. Look at what they rated highly that you haven't seen yet. Recommend that to you.
- **Item-Based CF**: Find items that are frequently bought together by the same users. If you buy Item X, recommend Item Y because everyone who buys X also buys Y.

## Key Concepts
- **Memory-Based CF**: Uses the entire user-item matrix directly to calculate similarities (nearest neighbors). It's simple but computationally expensive at scale.
- **Model-Based CF**: Uses machine learning to compress the matrix and find underlying "latent factors" (hidden patterns). Matrix Factorization and SVD are the prime examples.
- **Latent Factors**: Hidden variables. For example, the algorithm might discover a factor that strongly corresponds to "amount of action" in a movie, without ever being explicitly told what an action movie is.

## Step-by-Step Explanation: User-Based CF
1. **Take the Utility Matrix**: Rows are Users, Columns are Items.
2. **Calculate User Similarity**: Calculate the Cosine Similarity or Pearson Correlation between every pair of users.
3. **Find Neighbors**: For Target User, find the top K users who are most similar.
4. **Predict Rating**: To guess Target User's rating for Movie X, take the weighted average of the ratings given to Movie X by the K neighbors.
5. **Recommend**: Recommend the items with the highest predicted ratings.

## Important Parameters / Options / Settings
- **Similarity Metric**: Pearson Correlation is often preferred over Cosine Similarity for User-Based CF because it handles "tough graders" (users who always rate low) vs "easy graders" by centering the mean.
- **K (Number of Neighbors)**: How many similar users/items to consider. Too few = noisy recommendations. Too many = blends into the popularity baseline.
- **Model-Based Factors**: In Matrix Factorization, the number of latent factors (e.g., `n_factors=50`) controls the complexity of the model.

## Output / Result Interpretation
A list of recommended items or a predicted specific rating (e.g., predicting that User A will give Movie B 4.2 stars).

## Real-World Uses
- **Amazon**: Famous for pioneering Item-Item Collaborative Filtering.
- **Netflix Prize**: The famous $1 million prize was won by teams using Matrix Factorization techniques (SVD) to predict user ratings.

## Advantages
- **No Domain Knowledge Required**: You don't need to tag genres, extract text, or know anything about the items.
- **Serendipity**: Can recommend surprisingly good, unrelated items ("I didn't know I wanted a waffle maker, but users who bought my laptop also bought this waffle maker!").

## Limitations
- **Cold Start Problem**: Fails completely for new users (no history to find neighbors) and new items (nobody has rated it yet).
- **Sparsity**: If the matrix is 99.99% empty, it's very hard to find overlapping users who rated the same items.
- **Scalability**: Memory-based CF is too slow for millions of users (calculating pairwise similarity of 100M users is computationally infeasible in real-time).

## Common Mistakes
- **Using User-Based CF for E-commerce**: Users change tastes over time, and comparing millions of users is slow. Item-Item CF is usually much better for e-commerce.
- **Ignoring the Cold Start**: Recommending random items to new users. You must fall back to a Baseline or Content-Based system for new users.

## Related Methods
- Matrix Factorization (SVD, ALS)
- Deep Learning Recommenders (Neural CF)

## Code References
- `code/example-01-basic.py` — Manual User-User Collaborative Filtering.
- `code/example-02-intermediate.py` — Manual Item-Item Collaborative Filtering.
- `code/example-03-real-world.py` — Using the `Surprise` library for KNN-based CF.
- `code/example-04-advanced.py` — Matrix Factorization (SVD) using `Surprise`.
