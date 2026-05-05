# Interview Questions: Evaluation of Recommenders

## Beginner Questions
1. **Why is Accuracy not a good metric for a Recommender System?**
   *Answer*: Because the User-Item matrix is 99% zeros (unrated items). If a model predicts that the user won't interact with any item, it achieves 99% accuracy, but it recommends absolutely nothing.
2. **What does "RMSE" stand for and what does it measure in this context?**
   *Answer*: Root Mean Squared Error. It measures the average difference between the predicted rating (e.g., 4.5 stars) and the actual rating (e.g., 3.0 stars) across all predictions.

## Conceptual Questions
3. **What is the difference between Prediction Metrics and Ranking Metrics?**
   *Answer*: Prediction metrics (RMSE, MAE) evaluate how closely the model guesses the exact rating. Ranking metrics (Precision@K, Hit Ratio, NDCG) evaluate whether the items the user *actually* wants to see are placed at the very top of the recommendation list.
4. **Explain what NDCG is trying to measure.**
   *Answer*: Normalized Discounted Cumulative Gain measures the quality of a ranking. It assumes that an item placed at rank #1 is much more valuable to the user than the exact same item placed at rank #10, and discounts the score logarithmically the further down the list you go.

## Practical Questions
5. **You improve your model's RMSE from 0.85 to 0.70 offline, but when you deploy it, Click-Through Rate drops by 5%. Why might this happen?**
   *Answer*: Optimizing for RMSE often causes the model to recommend very "safe", universally well-rated items (like classic movies) to everyone. While these predictions are mathematically accurate, they are boring to users who want personalized, novel recommendations, leading to fewer clicks.
6. **How would you perform a Leave-One-Out evaluation to calculate Hit Ratio@10?**
   *Answer*: For each user, I would take one item they interacted with and set it aside as the target. I would then take 99 items they never interacted with. I would ask the model to score all 100 items and sort them. If the true target item appears in the Top 10 ranks, I record a Hit (1). Otherwise, a Miss (0). The average across all users is the Hit Ratio@10.
