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


---

## Method Options and Properties: Collaborative Filtering

Collaborative Filtering can be implemented manually using Pandas/Numpy for learning purposes, but in practice, specialized libraries like `Surprise` or `Implicit` are used.

### 1. Manual Implementation (Pandas)

#### `pandas.DataFrame.corr(method='pearson')`
- **Purpose**: Calculates the Pearson correlation coefficient between columns.
- **Usage in CF**: Used in Item-Item CF to find how correlated the ratings of Item A are with Item B across all users.
- **Why Pearson?**: It automatically handles mean-centering. It accounts for users who generally rate things higher or lower than average.

### 2. The `Surprise` Library
Scikit-Learn does not have built-in support for recommender systems. The `Surprise` library (Simple Python RecommendatIon System Engine) is the standard for teaching and basic model-based CF.

#### `surprise.Dataset.load_from_df()`
- **Purpose**: Loads a Pandas DataFrame into the format required by Surprise.
- **Syntax**: 
  ```python
  from surprise import Reader, Dataset
  reader = Reader(rating_scale=(1, 5))
  data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)
  ```
- **Constraint**: The DataFrame *must* have exactly 3 columns in the specific order: user, item, rating.

#### `surprise.prediction_algorithms.knns.KNNBasic`
- **Purpose**: Memory-based collaborative filtering (User-User or Item-Item).
- **Options**:
  - `sim_options = {'name': 'cosine', 'user_based': True}`: Does User-User Cosine similarity.
  - `sim_options = {'name': 'pearson', 'user_based': False}`: Does Item-Item Pearson correlation.
  - `k`: The max number of neighbors to consider (default 40).

#### `surprise.prediction_algorithms.matrix_factorization.SVD`
- **Purpose**: Model-based collaborative filtering using Singular Value Decomposition (Matrix Factorization).
- **Parameters**:
  - `n_factors`: Number of latent factors (default 100). Higher means more complex model, but risks overfitting.
  - `n_epochs`: Number of iterations of the SGD procedure (default 20).
  - `lr_all`: Learning rate (default 0.005).
  - `reg_all`: Regularization term to prevent overfitting (default 0.02).

### 3. Workflow for Prediction

#### `model.fit(trainset)`
- **Purpose**: Trains the algorithm on the utility matrix.

#### `model.predict(uid, iid)`
- **Purpose**: Predicts the rating a specific user (`uid`) would give a specific item (`iid`).
- **Return Type**: A `Prediction` object containing the estimated rating (`est`).

### Common Mistakes
- **Loading data incorrectly into Surprise**: Forgetting the `Reader` object or passing columns in the wrong order.
- **Using Memory-Based CF on huge datasets**: `KNNBasic` will run out of memory or take hours if you have millions of rows. Switch to `SVD`.

---

## Code Examples: Collaborative Filtering

These examples walk through both Memory-Based and Model-Based Collaborative Filtering.

### Code References

- **`code/example-01-basic.py`** — Manual User-User Collaborative Filtering. Shows the math behind finding user neighbors and predicting a rating based on their averages.
- **`code/example-02-intermediate.py`** — Manual Item-Item Collaborative Filtering using Pandas `corr()`. This is how early Amazon recommendations worked.
- **`code/example-03-real-world.py`** — Using the `Surprise` library to perform K-Nearest Neighbors Collaborative Filtering. This is how it's done professionally rather than writing manual loops.
- **`code/example-04-advanced.py`** — Matrix Factorization (SVD) using `Surprise`. This is the Model-Based approach that won the Netflix Prize.

---

## Practice: Collaborative Filtering

### Exercise 1: User-User vs Item-Item
1. Define User-User Collaborative Filtering in your own words.
2. Define Item-Item Collaborative Filtering in your own words.
3. Why is Item-Item generally preferred over User-User for large e-commerce websites like Amazon?

### Exercise 2: Manual Prediction
You are User A. You want to know if you will like Movie X.
You are most similar to User B (Similarity 0.9) and User C (Similarity 0.8).
- User B gave Movie X a rating of 5.
- User C gave Movie X a rating of 4.

Calculate the predicted rating for User A using a simple weighted average:
`(Sim_B * Rating_B + Sim_C * Rating_C) / (Sim_B + Sim_C)`

### Exercise 3: Python Coding (Surprise Library)
Using the `Surprise` library and its built-in `ml-100k` dataset:
1. Load the dataset.
2. Split the data into a training set and testing set.
3. Train an SVD model with `n_factors=50`.
4. Predict the rating that User '196' would give to Item '302'.

### Exercise 4: The Cold Start Problem
A new user signs up for your movie streaming service. You are using an SVD Matrix Factorization model.
1. What will the model predict for this user?
2. How can you architect your system to handle this? (Hint: Think about hybrid systems or baselines).

---

## Interview Questions: Collaborative Filtering

### Beginner Questions
1. **Explain Collaborative Filtering in simple terms.**
   *Answer*: It's a method that makes recommendations based on the past behavior of users. If two users had similar tastes in the past, the system assumes they will have similar tastes in the future. It relies on the "wisdom of the crowd."
2. **What is the difference between Memory-Based and Model-Based CF?**
   *Answer*: Memory-based CF uses the entire raw User-Item matrix to compute similarities (nearest neighbors) every time. Model-based CF uses machine learning (like Matrix Factorization) to compress the data and learn underlying patterns, making predictions much faster at scale.

### Conceptual Questions
3. **Why do large companies prefer Item-Item Collaborative Filtering over User-User?**
   *Answer*: 
   1. The number of items is usually much smaller than the number of users, making the similarity matrix smaller.
   2. Item characteristics are more static over time, whereas user tastes can change rapidly. 
   3. It allows for pre-computing the item similarities offline.
4. **What are "Latent Factors" in Matrix Factorization (SVD)?**
   *Answer*: Latent factors are hidden features discovered by the math. The algorithm might group items together without knowing why. For example, it might discover a factor that strongly corresponds to "comedy movies," even though no one explicitly labeled the movies as comedies.

### Practical Questions
5. **How does Matrix Factorization solve the sparsity problem?**
   *Answer*: It breaks down the massive, mostly empty User-Item matrix into two smaller, dense matrices (User Factors and Item Factors). Multiplying these smaller matrices together reconstructs the original matrix but fills in all the empty cells with predicted values.
6. **A user has rated only 1 item on your platform. Which approach is better: User-User CF or Item-Item CF?**
   *Answer*: Item-Item CF. With only 1 rating, it's impossible to find "similar users" accurately (User-User). However, if you know the item they rated, you can immediately recommend the most similar items to that specific item (Item-Item).

---

## Python Code Examples

### `example-01-basic.py`

```python
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Example 01: Manual User-User Collaborative Filtering
# Predicting a rating based on similar users.

print("--- Manual User-User CF ---")

# 1. Create a Utility Matrix (Users as rows, Movies as columns)
# 0 means unrated.
data = {
    'Matrix':    [5, 4, 1, 0],
    'Inception': [5, 5, 2, 0],
    'Titanic':   [1, 2, 5, 4],
    'Notebook':  [0, 1, 4, 5]
}
users = ['Alice', 'Bob', 'Charlie', 'David']
df = pd.DataFrame(data, index=users)

print("Utility Matrix:")
print(df)
print("\n")

# 2. Calculate User Similarity
# We compute the cosine similarity between the rows (users)
user_sim = cosine_similarity(df)
user_sim_df = pd.DataFrame(user_sim, index=users, columns=users)

print("User Similarity Matrix:")
print(user_sim_df.round(2))
print("\n")

# 3. Predict David's rating for 'Matrix'
# David hasn't watched the Matrix. Who is similar to David?
david_sims = user_sim_df.loc['David'].drop('David') # Drop self
print("David's similarity to others:")
print(david_sims.round(2))

# Charlie is the most similar to David (0.83). Alice and Bob are 0.
# Let's predict David's rating using a weighted average of everyone else's ratings for 'Matrix'

matrix_ratings = df['Matrix'].drop('David') # Everyone's rating for Matrix except David

# Weighted Average Formula: Sum(Similarity * Rating) / Sum(Similarities)
# Note: In real math we'd ignore users with a 0 similarity or 0 rating, but for this simple example:
numerator = sum(david_sims * matrix_ratings)
denominator = sum(david_sims[matrix_ratings > 0]) # Only divide by similarities of people who actually rated it

predicted_rating = numerator / denominator

print(f"\nPredicted rating for David on 'Matrix': {predicted_rating:.2f} stars")
# Since Charlie gave it a 1, and Charlie is David's only similar neighbor, it predicts 1.0!
```

### `example-02-intermediate.py`

```python
import pandas as pd

# Example 02: Manual Item-Item Collaborative Filtering
# "Customers who bought this item also bought..."
# We use Pearson Correlation to handle mean-centering (hard vs easy graders).

print("--- Manual Item-Item CF ---")

# 1. Create Utility Matrix
# NaN means unrated. Pearson correlation ignores NaNs automatically.
data = {
    'Matrix':    [5, 4, 1, float('nan'), 5],
    'Inception': [4, 5, 2, 1, 4],
    'Titanic':   [1, 2, 5, 4, 2],
    'Notebook':  [1, 2, 4, 5, 1],
    'Avatar':    [5, 5, 1, 2, 4]
}
users = ['U1', 'U2', 'U3', 'U4', 'U5']
df = pd.DataFrame(data, index=users)

print("Utility Matrix:")
print(df)
print("\n")

# 2. Calculate Item Similarity using Pearson Correlation
# df.corr() computes correlation between COLUMNS (Items)
item_sim = df.corr(method='pearson')

print("Item-Item Pearson Correlation Matrix:")
print(item_sim.round(2))
print("\n")

# 3. Recommend items
def recommend_item_to_item(movie_name, sim_matrix, top_n=2):
    # Get the column for the target movie
    similar_scores = sim_matrix[movie_name]
    
    # Sort descending
    similar_scores = similar_scores.sort_values(ascending=False)
    
    # Drop the movie itself (correlation 1.0)
    similar_scores = similar_scores.drop(movie_name)
    
    print(f"Because you watched {movie_name}, we recommend:")
    print(similar_scores.head(top_n).round(2))
    print("")

recommend_item_to_item('Matrix', item_sim)
recommend_item_to_item('Titanic', item_sim)

# Notice how highly Matrix correlates with Avatar and Inception!
```

### `example-03-real-world.py`

```python
# Example 03: Professional CF using Surprise Library (Memory-Based)
# In reality, you don't write manual math loops. You use specialized libraries.
# NOTE: To run this, you must install surprise: `pip install scikit-surprise`

try:
    from surprise import Dataset, Reader, KNNBasic
    from surprise.model_selection import train_test_split
    from surprise import accuracy
except ImportError:
    print("Please install surprise using: pip install scikit-surprise")
    exit()

print("--- KNN Collaborative Filtering with Surprise ---")

# 1. Load a built-in dataset (MovieLens 100k)
# This downloads a famous dataset of 100,000 movie ratings.
print("Loading MovieLens 100k dataset...")
data = Dataset.load_builtin('ml-100k')

# 2. Split into training and testing sets
trainset, testset = train_test_split(data, test_size=0.25)

# 3. Define the Algorithm
# We will use Item-Item Collaborative filtering using Cosine Similarity
sim_options = {
    'name': 'cosine',
    'user_based': False  # False means Item-Item. True means User-User.
}

# KNNBasic is standard memory-based nearest neighbors
algo = KNNBasic(sim_options=sim_options)

# 4. Train the algorithm on the trainset
print("Training model...")
algo.fit(trainset)

# 5. Predict on the testset and evaluate
print("Testing model...")
predictions = algo.test(testset)

# 6. Calculate RMSE (Root Mean Squared Error)
rmse = accuracy.rmse(predictions)
print(f"Model RMSE: {rmse:.4f}")

# 7. Make a specific prediction
# Predict what User '196' would rate Item '302'
uid = str(196)  # User id (as string because ml-100k uses strings)
iid = str(302)  # Item id
pred = algo.predict(uid, iid)

print("\nSpecific Prediction:")
print(f"User {uid} predicting Item {iid}")
print(f"Estimated rating: {pred.est:.2f}")
```

### `example-04-advanced.py`

```python
# Example 04: Matrix Factorization (Model-Based CF)
# This is the algorithm (SVD) that won the $1 Million Netflix Prize.
# NOTE: To run this, you must install surprise: `pip install scikit-surprise`

try:
    from surprise import Dataset, SVD
    from surprise.model_selection import cross_validate
except ImportError:
    print("Please install surprise using: pip install scikit-surprise")
    exit()

print("--- Matrix Factorization (SVD) with Surprise ---")

# 1. Load the dataset
data = Dataset.load_builtin('ml-100k')

# 2. Initialize the SVD algorithm
# n_factors = 50 means we are compressing the huge matrix into 50 "hidden" latent factors.
algo = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)

# 3. Evaluate using 5-Fold Cross Validation
# This automatically splits the data, trains, and tests 5 times to give a robust metric.
print("Running 5-fold cross validation...")
results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# 4. Print Average RMSE
print(f"\nAverage RMSE across 5 folds: {results['test_rmse'].mean():.4f}")

# Notice how the RMSE for SVD is usually lower (better) than the RMSE for KNNBasic
# in the previous example. Model-based approaches usually outperform memory-based approaches
# and they are MUCH faster to query in production.

# 5. Train on the whole dataset to put into "production"
trainset = data.build_full_trainset()
algo.fit(trainset)

# Predict User 196 rating Item 302
uid = str(196)
iid = str(302)
pred = algo.predict(uid, iid)

print(f"\nProduction Prediction for User {uid} on Item {iid}: {pred.est:.2f}")
```
