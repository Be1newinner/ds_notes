import pandas as pd

# Example 02: Building a Popularity-Based Baseline
# A Popularity baseline is the simplest recommendation engine.
# It just recommends the highest-rated or most frequently bought items to everyone.

print("--- Popularity-Based Recommender ---")

# 1. Mock dataset of e-commerce product ratings
data = {
    'user_id': ['U1', 'U2', 'U3', 'U1', 'U4', 'U5', 'U2', 'U6', 'U1', 'U3'],
    'product': ['Laptop', 'Laptop', 'Laptop', 'Mouse', 'Mouse', 'Mouse', 'Keyboard', 'Keyboard', 'Monitor', 'Monitor'],
    'rating': [5, 4, 5, 5, 1, 2, 4, 5, 5, 4]
}

df = pd.DataFrame(data)

# 2. Calculate average rating and total number of ratings for each product
# We group by product, and aggregate the 'rating' column calculating 'mean' and 'count'
product_stats = df.groupby('product')['rating'].agg(['mean', 'count']).reset_index()
print("Product Statistics:")
print(product_stats)
print("\n")

# 3. Build the recommendation logic
# Problem: The 'Mouse' has 3 ratings but a terrible mean (2.6).
# A product might have a 5.0 rating, but only 1 review. We shouldn't recommend that over a 4.8 with 1000 reviews.

# Let's recommend top products based on average rating, 
# BUT they must have at least 2 reviews (setting a minimum threshold).
min_reviews_threshold = 2

# Filter out items with too few reviews
qualified_products = product_stats[product_stats['count'] >= min_reviews_threshold]

# Sort by rating descending
top_products = qualified_products.sort_values(by='mean', ascending=False)

print(f"Top Recommended Products (Minimum {min_reviews_threshold} reviews):")
print(top_products[['product', 'mean', 'count']])

# This baseline is what we show to "Cold Start" users!
