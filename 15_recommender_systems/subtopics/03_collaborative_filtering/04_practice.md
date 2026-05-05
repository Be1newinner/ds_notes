# Practice: Collaborative Filtering

## Exercise 1: User-User vs Item-Item
1. Define User-User Collaborative Filtering in your own words.
2. Define Item-Item Collaborative Filtering in your own words.
3. Why is Item-Item generally preferred over User-User for large e-commerce websites like Amazon?

## Exercise 2: Manual Prediction
You are User A. You want to know if you will like Movie X.
You are most similar to User B (Similarity 0.9) and User C (Similarity 0.8).
- User B gave Movie X a rating of 5.
- User C gave Movie X a rating of 4.

Calculate the predicted rating for User A using a simple weighted average:
`(Sim_B * Rating_B + Sim_C * Rating_C) / (Sim_B + Sim_C)`

## Exercise 3: Python Coding (Surprise Library)
Using the `Surprise` library and its built-in `ml-100k` dataset:
1. Load the dataset.
2. Split the data into a training set and testing set.
3. Train an SVD model with `n_factors=50`.
4. Predict the rating that User '196' would give to Item '302'.

## Exercise 4: The Cold Start Problem
A new user signs up for your movie streaming service. You are using an SVD Matrix Factorization model.
1. What will the model predict for this user?
2. How can you architect your system to handle this? (Hint: Think about hybrid systems or baselines).
