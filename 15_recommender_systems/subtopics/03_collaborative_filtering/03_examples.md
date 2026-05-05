# Code Examples: Collaborative Filtering

These examples walk through both Memory-Based and Model-Based Collaborative Filtering.

## Code References

- **`code/example-01-basic.py`** — Manual User-User Collaborative Filtering. Shows the math behind finding user neighbors and predicting a rating based on their averages.
- **`code/example-02-intermediate.py`** — Manual Item-Item Collaborative Filtering using Pandas `corr()`. This is how early Amazon recommendations worked.
- **`code/example-03-real-world.py`** — Using the `Surprise` library to perform K-Nearest Neighbors Collaborative Filtering. This is how it's done professionally rather than writing manual loops.
- **`code/example-04-advanced.py`** — Matrix Factorization (SVD) using `Surprise`. This is the Model-Based approach that won the Netflix Prize.
