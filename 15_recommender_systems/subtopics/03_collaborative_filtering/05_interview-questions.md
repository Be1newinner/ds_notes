# Interview Questions: Collaborative Filtering

## Beginner Questions
1. **Explain Collaborative Filtering in simple terms.**
   *Answer*: It's a method that makes recommendations based on the past behavior of users. If two users had similar tastes in the past, the system assumes they will have similar tastes in the future. It relies on the "wisdom of the crowd."
2. **What is the difference between Memory-Based and Model-Based CF?**
   *Answer*: Memory-based CF uses the entire raw User-Item matrix to compute similarities (nearest neighbors) every time. Model-based CF uses machine learning (like Matrix Factorization) to compress the data and learn underlying patterns, making predictions much faster at scale.

## Conceptual Questions
3. **Why do large companies prefer Item-Item Collaborative Filtering over User-User?**
   *Answer*: 
   1. The number of items is usually much smaller than the number of users, making the similarity matrix smaller.
   2. Item characteristics are more static over time, whereas user tastes can change rapidly. 
   3. It allows for pre-computing the item similarities offline.
4. **What are "Latent Factors" in Matrix Factorization (SVD)?**
   *Answer*: Latent factors are hidden features discovered by the math. The algorithm might group items together without knowing why. For example, it might discover a factor that strongly corresponds to "comedy movies," even though no one explicitly labeled the movies as comedies.

## Practical Questions
5. **How does Matrix Factorization solve the sparsity problem?**
   *Answer*: It breaks down the massive, mostly empty User-Item matrix into two smaller, dense matrices (User Factors and Item Factors). Multiplying these smaller matrices together reconstructs the original matrix but fills in all the empty cells with predicted values.
6. **A user has rated only 1 item on your platform. Which approach is better: User-User CF or Item-Item CF?**
   *Answer*: Item-Item CF. With only 1 rating, it's impossible to find "similar users" accurately (User-User). However, if you know the item they rated, you can immediately recommend the most similar items to that specific item (Item-Item).
