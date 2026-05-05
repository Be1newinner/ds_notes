# Interview Questions: Problem Framing

## Beginner Questions
1. **What is the difference between explicit and implicit feedback?**
   *Answer*: Explicit feedback is a direct rating from a user (e.g., 5 stars). Implicit feedback is inferred from behavior (e.g., clicks, views, purchases).
2. **What is a User-Item Utility Matrix?**
   *Answer*: A 2D matrix where rows represent users, columns represent items, and values represent the interaction (rating or boolean) between them.

## Conceptual Questions
3. **What is the "Cold Start" problem in recommender systems?**
   *Answer*: It is the challenge of making recommendations for a brand new user who has no history, or recommending a brand new item that no one has interacted with yet.
4. **Why is the utility matrix usually described as "sparse"?**
   *Answer*: Because the vast majority of users have only interacted with a tiny fraction of the total available items. A matrix might be 99.9% zeros/empty.

## Practical Questions
5. **How would you build a recommender system for a brand new app launching tomorrow with zero user history?**
   *Answer*: I would use a Popularity-based baseline (recommending universally popular items or editor's picks) or Content-based filtering (recommending items based on user demographic info provided during sign-up).
6. **If you have a dataset of 1 million users and 100,000 products, can you use Pandas `pivot_table` to build the utility matrix?**
   *Answer*: Usually no. A 1 million x 100,000 matrix contains 100 billion cells. Storing that as a standard dense dataframe in memory will crash most machines. You must use a Sparse Matrix format (like `scipy.sparse`).
