# Practice: Evaluation of Recommenders

## Exercise 1: RMSE Calculation
You have the following true ratings and predicted ratings:
- Item 1: True=5, Pred=4
- Item 2: True=2, Pred=2
- Item 3: True=1, Pred=3

1. Calculate the MAE (Mean Absolute Error).
2. Calculate the MSE (Mean Squared Error).
3. Calculate the RMSE (Root Mean Squared Error).

## Exercise 2: Precision@K and Recall@K
A user has 4 movies in their "Favorites" list: [A, B, C, D].
Your model generates a Top 5 recommendation list for this user: [X, B, Y, Z, D].

1. Calculate Precision@5.
2. Calculate Recall@5.

## Exercise 3: The Hit Ratio Concept
You are doing a "Leave-One-Out" evaluation. You hid a movie the user watched (`Movie_Target`). You mixed `Movie_Target` with 99 random movies the user has never watched. Your model ranks these 100 movies.
`Movie_Target` appears at rank #12.

1. If you are calculating Hit Ratio for K=10 (Top 10), is this a Hit (1) or a Miss (0)?
2. If you change your UI to show a Top 20 list (K=20), is it a Hit or a Miss?

## Exercise 4: Python Coding Challenge
Write a Python function `precision_at_k(recommended_list, actual_list, k)` that takes a list of recommended item IDs, a list of actual liked item IDs, and an integer K, and returns the Precision@K metric.
