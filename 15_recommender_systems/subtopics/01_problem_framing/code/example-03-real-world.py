import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

# Example 03: Sparsity and Memory Management (Real World)
# In the real world, pivot_table will crash your computer if you have millions of rows.
# We must use sparse matrices.

print("--- Matrix Sparsity and Scipy Sparse Matrices ---")

# 1. Create a larger, mostly empty dataset (simulating reality)
# 10 users, 10 items, but only 15 interactions (out of 100 possible)
users = np.random.randint(0, 10, 15)
items = np.random.randint(0, 10, 15)
ratings = np.random.randint(1, 6, 15)

df = pd.DataFrame({'user_id': users, 'item_id': items, 'rating': ratings})
# Drop duplicates just in case random generated the same user-item pair
df = df.drop_duplicates(subset=['user_id', 'item_id'])

print(f"Total interactions recorded: {len(df)}")

# 2. Calculate Sparsity mathematically BEFORE creating the matrix
total_possible_users = 10
total_possible_items = 10
total_possible_interactions = total_possible_users * total_possible_items

actual_interactions = len(df)
sparsity = 1.0 - (actual_interactions / total_possible_interactions)

print(f"Matrix Sparsity: {sparsity * 100:.2f}% empty cells\n")

# 3. Create a Sparse Matrix instead of a Pandas Pivot Table
# We map users and items to matrix indices
user_cat = df['user_id'].astype('category').cat.codes
item_cat = df['item_id'].astype('category').cat.codes

# Create Compressed Sparse Row matrix
sparse_utility_matrix = csr_matrix((df['rating'], (user_cat, item_cat)), 
                                   shape=(total_possible_users, total_possible_items))

print("Sparse Matrix representation (only stores non-zero values):")
print(sparse_utility_matrix)

# Notice it prints coordinate pairs (row, col) -> value
# This takes a tiny fraction of the RAM compared to a Pandas DataFrame full of zeros.
