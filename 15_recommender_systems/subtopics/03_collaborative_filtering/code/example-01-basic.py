import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Example 01: Manual User-User Collaborative Filtering
# Predicting a rating based on similar users.

print("--- Manual User-User CF ---")

# 1. Create a Utility Matrix (Users as rows, Movies as columns)
# 0 means unrated.
data = {
    'Matrix':    [5, 4, 1, 0],
    'Inception': [5, 5, 2, 0],
    'Titanic':   [1, 2, 5, 4],
    'Notebook':  [0, 1, 4, 5]
}
users = ['Alice', 'Bob', 'Charlie', 'David']
df = pd.DataFrame(data, index=users)

print("Utility Matrix:")
print(df)
print("\n")

# 2. Calculate User Similarity
# We compute the cosine similarity between the rows (users)
user_sim = cosine_similarity(df)
user_sim_df = pd.DataFrame(user_sim, index=users, columns=users)

print("User Similarity Matrix:")
print(user_sim_df.round(2))
print("\n")

# 3. Predict David's rating for 'Matrix'
# David hasn't watched the Matrix. Who is similar to David?
david_sims = user_sim_df.loc['David'].drop('David') # Drop self
print("David's similarity to others:")
print(david_sims.round(2))

# Charlie is the most similar to David (0.83). Alice and Bob are 0.
# Let's predict David's rating using a weighted average of everyone else's ratings for 'Matrix'

matrix_ratings = df['Matrix'].drop('David') # Everyone's rating for Matrix except David

# Weighted Average Formula: Sum(Similarity * Rating) / Sum(Similarities)
# Note: In real math we'd ignore users with a 0 similarity or 0 rating, but for this simple example:
numerator = sum(david_sims * matrix_ratings)
denominator = sum(david_sims[matrix_ratings > 0]) # Only divide by similarities of people who actually rated it

predicted_rating = numerator / denominator

print(f"\nPredicted rating for David on 'Matrix': {predicted_rating:.2f} stars")
# Since Charlie gave it a 1, and Charlie is David's only similar neighbor, it predicts 1.0!
