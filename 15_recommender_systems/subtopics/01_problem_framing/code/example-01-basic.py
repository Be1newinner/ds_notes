import pandas as pd

# Example 01: Building a Basic Utility Matrix
# This example demonstrates how to take a list of transactions (long format)
# and convert it into a User-Item Utility Matrix (wide format).

print("--- Basic Utility Matrix ---")

# 1. Create a mock dataset of user ratings (Long Format)
data = {
    'user_id': ['User_A', 'User_A', 'User_B', 'User_B', 'User_C', 'User_D'],
    'movie_id': ['Matrix', 'Inception', 'Matrix', 'Avatar', 'Inception', 'Avatar'],
    'rating': [5, 4, 3, 5, 2, 4]
}

df = pd.DataFrame(data)
print("Raw Transaction Data (Long Format):")
print(df)
print("\n")

# 2. Convert to Utility Matrix using pivot_table
# index = rows (users)
# columns = columns (items)
# values = data inside cells (ratings)
# fill_value = what to put if the user didn't rate the item
utility_matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating', fill_value=0)

print("User-Item Utility Matrix (Wide Format):")
print(utility_matrix)

# Notice how the matrix is created. If User_C didn't rate Avatar, it gets a 0.
