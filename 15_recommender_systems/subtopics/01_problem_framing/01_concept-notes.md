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
