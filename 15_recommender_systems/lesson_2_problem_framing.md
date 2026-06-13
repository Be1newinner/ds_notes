# Problem Framing in Recommender Systems

## Learning Objective
What students should understand after this lesson:
- The fundamental goal of a recommendation system.
- The difference between explicit and implicit feedback.
- What a User-Item Utility Matrix is.
- The "Cold Start" problem.
- How to build a simple baseline (Popularity-Based Recommender).

## What Is This Topic?
Problem Framing is about understanding how to set up data for a recommendation system. Before we can predict what a user wants, we have to structure the data into a format that algorithms can understand, usually involving users, items, and the interactions between them.

## Why This Topic Matters
If you don't frame the problem correctly, even the most advanced AI will fail. You must decide whether you are predicting a rating (e.g., 1-5 stars) or predicting an action (e.g., click or buy), as this completely changes the approach and the metrics used.

## Core Intuition
Imagine a giant spreadsheet where rows are users (you, your friends) and columns are items (movies, books). The cells contain the ratings given. This spreadsheet is mostly empty because no one has watched every movie. A recommender system's job is simply to *fill in the blank cells* with good guesses.

## Key Concepts
- **Utility Matrix**: The User-Item matrix containing ratings or interactions.
- **Explicit Feedback**: When a user directly tells you their preference (e.g., 5-star rating, thumbs up/down).
- **Implicit Feedback**: When you infer preference from user behavior (e.g., clicks, watch time, purchase history).
- **Sparsity**: The utility matrix is highly sparse (most users haven't interacted with most items).
- **Cold Start Problem**: How do you recommend items to a brand new user? How do you recommend a brand new item to anyone?

## Step-by-Step Explanation
1. **Identify the Users and Items**: Define who is receiving the recommendation and what is being recommended.
2. **Gather Interactions**: Collect data on how users have interacted with items (clicks, ratings).
3. **Build the Utility Matrix**: Structure the data into a matrix format.
4. **Define the Goal**: Are we trying to predict the exact rating, or just rank the top 10 items to show the user?
5. **Establish a Baseline**: Always start with a non-personalized baseline, like recommending the most popular items to everyone.

## Important Parameters / Options / Settings
- **Feedback Type**: Explicit vs Implicit.
- **Sparsity Level**: `1 - (number of interactions) / (total possible interactions)`. Often > 99%.

## Output / Result Interpretation
The output of framing is a structured dataset and a clear baseline performance. The Popularity Baseline simply outputs the top N most frequently interacted-with items.

## Real-World Uses
- Setting up the data pipeline for Netflix recommendations.
- E-commerce sites logging "clicks" and "add to cart" events to form implicit feedback matrices.

## Advantages of Baselines (Popularity)
- Extremely easy to build.
- Solves the new user Cold Start problem (just show them what's trending).
- Often surprisingly effective as a benchmark.

## Limitations of Baselines
- No personalization at all.
- Tends to create a "rich get richer" loop where popular items are recommended more, making them even more popular.

## Common Mistakes
- **Treating implicit feedback as explicit**: A click does not mean they liked it (they might have clicked by accident or disliked the item after viewing).
- **Ignoring the Cold Start problem**: Building a complex model that completely breaks when a new user signs up.

## Related Methods
- Content-Based Filtering (next topic)
- Collaborative Filtering (next topic)

## Code References
- `code/example-01-basic.py` — Building a basic utility matrix.
- `code/example-02-intermediate.py` — Building a Popularity-Based Recommender.
- `code/example-03-real-world.py` — Handling implicit vs explicit data with Pandas.


---

## Method Options and Properties: Problem Framing

While Problem Framing is largely conceptual, there are specific Pandas and Numpy methods used to construct and manipulate the Utility Matrix.

### 1. Creating the Utility Matrix

#### `pandas.DataFrame.pivot_table()`
- **Purpose**: Reshapes data from a long format (transactions/interactions) to a wide format (User-Item matrix).
- **Syntax**: `df.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0)`
- **Common Arguments**:
  - `index`: The column to use for rows (typically User ID).
  - `columns`: The column to use for columns (typically Item ID).
  - `values`: The column containing the interaction/rating.
  - `fill_value`: Value to replace missing combinations (usually 0, though in some algorithms NaN is kept).
  - `aggfunc`: Function to use if there are multiple interactions (e.g., `mean` or `sum`).
- **Return Type**: A new Pandas DataFrame.

#### `scipy.sparse.csr_matrix`
- **Purpose**: When utility matrices become too large for Pandas memory, they must be converted to sparse matrices.
- **Syntax**: `csr_matrix(df.values)`
- **Common Usage**: Scikit-learn and specialized recommender libraries require CSR (Compressed Sparse Row) matrices for efficiency.

### 2. Building a Popularity Baseline

#### `pandas.DataFrame.groupby()` and `agg()`
- **Purpose**: To calculate the popularity or average rating of items.
- **Syntax**: `df.groupby('item_id').agg({'rating': ['count', 'mean']})`
- **Common Workflow**:
  1. Group by item.
  2. Count the number of interactions.
  3. Calculate the average rating.
  4. Filter out items with very few interactions.
  5. Sort by rating or count descending.

#### `pandas.DataFrame.sort_values()`
- **Purpose**: To rank the items to generate the top N recommendations.
- **Syntax**: `df.sort_values(by='rating_count', ascending=False)`

### Common Mistakes
- **Memory Errors with `pivot_table`**: Calling `pivot_table` on millions of rows can cause a RAM crash. In real-world scenarios, sparse matrices (`scipy.sparse`) are required.
- **Not filling NaNs**: Machine learning algorithms cannot handle `NaN` values in the utility matrix. You must decide whether to fill them with 0, the user's mean, or the item's mean.

---

## Code Examples: Problem Framing

These examples show how to take raw transaction or rating data and frame it for recommendation systems, including building baselines.

### Code References

- **`code/example-01-basic.py`** — Shows how to build a basic User-Item Utility Matrix from a list of transactions using Pandas `pivot_table`.
- **`code/example-02-intermediate.py`** — Demonstrates building a Popularity-Based Recommender (a baseline model) taking both average rating and count into account.
- **`code/example-03-real-world.py`** — Shows how to calculate the sparsity of a matrix and convert it to a Scipy Sparse Matrix, which is necessary for real-world memory constraints.

---

## Practice: Problem Framing

### Exercise 1: Build a Utility Matrix
Given the following dataset of movie ratings:
- User A rated Movie 1 (5 stars)
- User A rated Movie 2 (4 stars)
- User B rated Movie 2 (2 stars)
- User B rated Movie 3 (5 stars)
- User C rated Movie 1 (1 star)

**Task:**
1. Manually draw the User-Item Utility Matrix.
2. What is the sparsity of this matrix? (Calculate the percentage of empty cells).

### Exercise 2: Identify Feedback Types
Classify the following as Explicit or Implicit feedback:
1. A user clicking "Add to Wishlist".
2. A user giving a 10/10 rating on IMDb.
3. A user watching 45 minutes of a 1-hour YouTube video.
4. A user leaving a written review saying "Terrible product".
5. A user skipping a song on Spotify after 10 seconds.

### Exercise 3: Python Coding
Load a dataset of e-commerce purchases (e.g., `User_ID`, `Product_ID`, `Price`).
1. Create a Popularity-Based baseline that recommends the top 5 products based purely on the number of times they were purchased.
2. Modify your baseline so that it only recommends products that have an average price greater than $50.

---

## Interview Questions: Problem Framing

### Beginner Questions
1. **What is the difference between explicit and implicit feedback?**
   *Answer*: Explicit feedback is a direct rating from a user (e.g., 5 stars). Implicit feedback is inferred from behavior (e.g., clicks, views, purchases).
2. **What is a User-Item Utility Matrix?**
   *Answer*: A 2D matrix where rows represent users, columns represent items, and values represent the interaction (rating or boolean) between them.

### Conceptual Questions
3. **What is the "Cold Start" problem in recommender systems?**
   *Answer*: It is the challenge of making recommendations for a brand new user who has no history, or recommending a brand new item that no one has interacted with yet.
4. **Why is the utility matrix usually described as "sparse"?**
   *Answer*: Because the vast majority of users have only interacted with a tiny fraction of the total available items. A matrix might be 99.9% zeros/empty.

### Practical Questions
5. **How would you build a recommender system for a brand new app launching tomorrow with zero user history?**
   *Answer*: I would use a Popularity-based baseline (recommending universally popular items or editor's picks) or Content-based filtering (recommending items based on user demographic info provided during sign-up).
6. **If you have a dataset of 1 million users and 100,000 products, can you use Pandas `pivot_table` to build the utility matrix?**
   *Answer*: Usually no. A 1 million x 100,000 matrix contains 100 billion cells. Storing that as a standard dense dataframe in memory will crash most machines. You must use a Sparse Matrix format (like `scipy.sparse`).

---

## Python Code Examples

### `example-01-basic.py`

```python
import pandas as pd

# Example 01: Building a Basic Utility Matrix
# This example demonstrates how to take a list of transactions (long format)
# and convert it into a User-Item Utility Matrix (wide format).

print("--- Basic Utility Matrix ---")

# 1. Create a mock dataset of user ratings (Long Format)
data = {
    'user_id': ['User_A', 'User_A', 'User_B', 'User_B', 'User_C', 'User_D'],
    'movie_id': ['Matrix', 'Inception', 'Matrix', 'Avatar', 'Inception', 'Avatar'],
    'rating': [5, 4, 3, 5, 2, 4]
}

df = pd.DataFrame(data)
print("Raw Transaction Data (Long Format):")
print(df)
print("\n")

# 2. Convert to Utility Matrix using pivot_table
# index = rows (users)
# columns = columns (items)
# values = data inside cells (ratings)
# fill_value = what to put if the user didn't rate the item
utility_matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating', fill_value=0)

print("User-Item Utility Matrix (Wide Format):")
print(utility_matrix)

# Notice how the matrix is created. If User_C didn't rate Avatar, it gets a 0.
```

### `example-02-intermediate.py`

```python
import pandas as pd

# Example 02: Building a Popularity-Based Baseline
# A Popularity baseline is the simplest recommendation engine.
# It just recommends the highest-rated or most frequently bought items to everyone.

print("--- Popularity-Based Recommender ---")

# 1. Mock dataset of e-commerce product ratings
data = {
    'user_id': ['U1', 'U2', 'U3', 'U1', 'U4', 'U5', 'U2', 'U6', 'U1', 'U3'],
    'product': ['Laptop', 'Laptop', 'Laptop', 'Mouse', 'Mouse', 'Mouse', 'Keyboard', 'Keyboard', 'Monitor', 'Monitor'],
    'rating': [5, 4, 5, 5, 1, 2, 4, 5, 5, 4]
}

df = pd.DataFrame(data)

# 2. Calculate average rating and total number of ratings for each product
# We group by product, and aggregate the 'rating' column calculating 'mean' and 'count'
product_stats = df.groupby('product')['rating'].agg(['mean', 'count']).reset_index()
print("Product Statistics:")
print(product_stats)
print("\n")

# 3. Build the recommendation logic
# Problem: The 'Mouse' has 3 ratings but a terrible mean (2.6).
# A product might have a 5.0 rating, but only 1 review. We shouldn't recommend that over a 4.8 with 1000 reviews.

# Let's recommend top products based on average rating, 
# BUT they must have at least 2 reviews (setting a minimum threshold).
min_reviews_threshold = 2

# Filter out items with too few reviews
qualified_products = product_stats[product_stats['count'] >= min_reviews_threshold]

# Sort by rating descending
top_products = qualified_products.sort_values(by='mean', ascending=False)

print(f"Top Recommended Products (Minimum {min_reviews_threshold} reviews):")
print(top_products[['product', 'mean', 'count']])

# This baseline is what we show to "Cold Start" users!
```

### `example-03-real-world.py`

```python
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

# Example 03: Sparsity and Memory Management (Real World)
# In the real world, pivot_table will crash your computer if you have millions of rows.
# We must use sparse matrices.

print("--- Matrix Sparsity and Scipy Sparse Matrices ---")

# 1. Create a larger, mostly empty dataset (simulating reality)
# 10 users, 10 items, but only 15 interactions (out of 100 possible)
users = np.random.randint(0, 10, 15)
items = np.random.randint(0, 10, 15)
ratings = np.random.randint(1, 6, 15)

df = pd.DataFrame({'user_id': users, 'item_id': items, 'rating': ratings})
# Drop duplicates just in case random generated the same user-item pair
df = df.drop_duplicates(subset=['user_id', 'item_id'])

print(f"Total interactions recorded: {len(df)}")

# 2. Calculate Sparsity mathematically BEFORE creating the matrix
total_possible_users = 10
total_possible_items = 10
total_possible_interactions = total_possible_users * total_possible_items

actual_interactions = len(df)
sparsity = 1.0 - (actual_interactions / total_possible_interactions)

print(f"Matrix Sparsity: {sparsity * 100:.2f}% empty cells\n")

# 3. Create a Sparse Matrix instead of a Pandas Pivot Table
# We map users and items to matrix indices
user_cat = df['user_id'].astype('category').cat.codes
item_cat = df['item_id'].astype('category').cat.codes

# Create Compressed Sparse Row matrix
sparse_utility_matrix = csr_matrix((df['rating'], (user_cat, item_cat)), 
                                   shape=(total_possible_users, total_possible_items))

print("Sparse Matrix representation (only stores non-zero values):")
print(sparse_utility_matrix)

# Notice it prints coordinate pairs (row, col) -> value
# This takes a tiny fraction of the RAM compared to a Pandas DataFrame full of zeros.
```
