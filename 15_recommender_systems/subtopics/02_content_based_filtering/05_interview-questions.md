# Interview Questions: Content-Based Filtering

## Beginner Questions
1. **Explain Content-Based Filtering in one sentence.**
   *Answer*: It recommends items to a user that are similar in features or content to items the user has liked in the past.
2. **What is Cosine Similarity?**
   *Answer*: It's a metric that measures the cosine of the angle between two vectors. It determines how similar two items are, ranging from 1 (identical) to 0 (completely different).

## Conceptual Questions
3. **Why do we use TF-IDF instead of simple word counts for text-based recommendations?**
   *Answer*: Because simple word counts give too much importance to frequent words like "the" or "is". TF-IDF penalizes words that appear across all documents, highlighting unique words that actually describe the specific content of an item.
4. **How does Content-Based Filtering solve the Item Cold Start problem?**
   *Answer*: Since it relies purely on item metadata (like genres, descriptions, authors), it does not need any user interaction history to understand the item. A brand new item can instantly be matched to users with similar tastes.

## Practical Questions
5. **What is a major disadvantage of Content-Based Filtering?**
   *Answer*: It suffers from "overspecialization" or the "filter bubble" effect. It can only recommend items similar to what the user has already seen. It cannot help a user discover entirely new genres or tastes outside their established profile.
6. **You have 1 million articles. Can you run `cosine_similarity()` on all of them to find recommendations?**
   *Answer*: No, computing the pairwise similarity of 1 million items results in a 1 trillion element matrix, which will cause an Out-Of-Memory (OOM) error. Instead, you would calculate similarity on the fly for a specific query, or use Approximate Nearest Neighbor (ANN) search algorithms like FAISS or Annoy.
