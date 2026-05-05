# Code Examples: Problem Framing

These examples show how to take raw transaction or rating data and frame it for recommendation systems, including building baselines.

## Code References

- **`code/example-01-basic.py`** — Shows how to build a basic User-Item Utility Matrix from a list of transactions using Pandas `pivot_table`.
- **`code/example-02-intermediate.py`** — Demonstrates building a Popularity-Based Recommender (a baseline model) taking both average rating and count into account.
- **`code/example-03-real-world.py`** — Shows how to calculate the sparsity of a matrix and convert it to a Scipy Sparse Matrix, which is necessary for real-world memory constraints.
