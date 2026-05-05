# Practice: Problem Framing

## Exercise 1: Build a Utility Matrix
Given the following dataset of movie ratings:
- User A rated Movie 1 (5 stars)
- User A rated Movie 2 (4 stars)
- User B rated Movie 2 (2 stars)
- User B rated Movie 3 (5 stars)
- User C rated Movie 1 (1 star)

**Task:**
1. Manually draw the User-Item Utility Matrix.
2. What is the sparsity of this matrix? (Calculate the percentage of empty cells).

## Exercise 2: Identify Feedback Types
Classify the following as Explicit or Implicit feedback:
1. A user clicking "Add to Wishlist".
2. A user giving a 10/10 rating on IMDb.
3. A user watching 45 minutes of a 1-hour YouTube video.
4. A user leaving a written review saying "Terrible product".
5. A user skipping a song on Spotify after 10 seconds.

## Exercise 3: Python Coding
Load a dataset of e-commerce purchases (e.g., `User_ID`, `Product_ID`, `Price`).
1. Create a Popularity-Based baseline that recommends the top 5 products based purely on the number of times they were purchased.
2. Modify your baseline so that it only recommends products that have an average price greater than $50.
