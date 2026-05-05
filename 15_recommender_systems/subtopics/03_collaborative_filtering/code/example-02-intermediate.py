import pandas as pd

# Example 02: Manual Item-Item Collaborative Filtering
# "Customers who bought this item also bought..."
# We use Pearson Correlation to handle mean-centering (hard vs easy graders).

print("--- Manual Item-Item CF ---")

# 1. Create Utility Matrix
# NaN means unrated. Pearson correlation ignores NaNs automatically.
data = {
    'Matrix':    [5, 4, 1, float('nan'), 5],
    'Inception': [4, 5, 2, 1, 4],
    'Titanic':   [1, 2, 5, 4, 2],
    'Notebook':  [1, 2, 4, 5, 1],
    'Avatar':    [5, 5, 1, 2, 4]
}
users = ['U1', 'U2', 'U3', 'U4', 'U5']
df = pd.DataFrame(data, index=users)

print("Utility Matrix:")
print(df)
print("\n")

# 2. Calculate Item Similarity using Pearson Correlation
# df.corr() computes correlation between COLUMNS (Items)
item_sim = df.corr(method='pearson')

print("Item-Item Pearson Correlation Matrix:")
print(item_sim.round(2))
print("\n")

# 3. Recommend items
def recommend_item_to_item(movie_name, sim_matrix, top_n=2):
    # Get the column for the target movie
    similar_scores = sim_matrix[movie_name]
    
    # Sort descending
    similar_scores = similar_scores.sort_values(ascending=False)
    
    # Drop the movie itself (correlation 1.0)
    similar_scores = similar_scores.drop(movie_name)
    
    print(f"Because you watched {movie_name}, we recommend:")
    print(similar_scores.head(top_n).round(2))
    print("")

recommend_item_to_item('Matrix', item_sim)
recommend_item_to_item('Titanic', item_sim)

# Notice how highly Matrix correlates with Avatar and Inception!
