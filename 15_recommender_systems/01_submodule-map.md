# Recommender Systems Submodule Map

This document outlines the subtopics covered in the Recommender Systems module, their purpose, and the recommended teaching sequence.

## 01. Problem Framing
- **Why it is taught**: Sets the stage. Students need to understand what data looks like for recommenders (utility matrices) and understand simple baselines before jumping into complex math.
- **Theory-heavy**: Yes. Explains implicit vs explicit feedback and the cold start problem.
- **Needs Code**: Yes. Simple baseline (Popularity-based recommender).
- **Needs Visual Explanation**: Yes. Show what a User-Item Utility Matrix looks like (sparse vs dense).
- **Needs Business Examples**: Yes. Amazon Top Sellers, Netflix Top 10.

## 02. Content-Based Filtering
- **Why it is taught**: Teaches students how to use item metadata (text, tags, genres) to make recommendations when user interaction data is sparse.
- **Theory-heavy**: Medium. Relies on similarity metrics (Cosine Similarity) and vectorization (TF-IDF).
- **Needs Code**: Yes. Building a movie recommender using plot descriptions or genres.
- **Needs Visual Explanation**: Yes. Show how an item vector is compared to a user profile vector.
- **Needs Business Examples**: Yes. Recommending news articles based on text similarity.

## 03. Collaborative Filtering
- **Why it is taught**: The core of modern recommendation. Teaches how to leverage the "wisdom of the crowd" without needing item metadata.
- **Theory-heavy**: High. Explains memory-based (User-User, Item-Item) and introduces model-based (Matrix Factorization / SVD).
- **Needs Code**: Yes. Implementing CF using Pandas or specialized libraries like `Surprise`.
- **Needs Visual Explanation**: Yes. Visualizing user-user similarity vs item-item similarity.
- **Needs Business Examples**: Yes. Amazon's classic item-to-item collaborative filtering.

## 04. Evaluation of Recommenders
- **Why it is taught**: Recommender evaluation is fundamentally different from standard classification. You evaluate *rankings* and *top-k* lists, not just absolute predictions.
- **Theory-heavy**: High. Explains RMSE, Precision@k, Recall@k, and Hit Ratio.
- **Needs Code**: Yes. Writing functions to calculate Hit Ratio or Precision@k.
- **Needs Visual Explanation**: Yes. Explaining what "@k" means visually.
- **Needs Business Examples**: Yes. Why Netflix cares more about the Top 10 than the 100th item.

## Recommended Order of Teaching
1. `01_problem_framing` - Start with the basics and utility matrices.
2. `02_content_based_filtering` - Use item features (easier intuition if students know NLP/Text Basics).
3. `03_collaborative_filtering` - Move to behavior-based recommendations.
4. `04_evaluation_of_recommenders` - Wrap up with how to measure success.
