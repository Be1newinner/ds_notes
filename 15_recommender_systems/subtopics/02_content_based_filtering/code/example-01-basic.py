import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example 01: Simple Tag-Based Recommender
# Recommending movies based purely on their Genre tags.

print("--- Simple Tag-Based Content Recommender ---")

# 1. Mock dataset of movies and their genres
data = {
    'movie_title': ['The Matrix', 'Inception', 'Toy Story', 'Finding Nemo', 'Terminator'],
    'genres': ['Action Sci-Fi', 'Action Sci-Fi Thriller', 'Animation Comedy Family', 'Animation Family', 'Action Sci-Fi']
}
df = pd.DataFrame(data)

# 2. Vectorize the text tags using CountVectorizer
# We use CountVectorizer because these are simple tags, not complex natural language sentences.
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['genres'])

print("Vocabulary Learned:")
print(cv.get_feature_names_out())
print("\nVectorized Matrix (Dense representation):")
print(count_matrix.toarray())
print("\n")

# 3. Calculate Cosine Similarity between all movies
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# 4. Create a recommendation function
def recommend(title, df, sim_matrix):
    # Get the index of the movie that matches the title
    idx = df[df['movie_title'] == title].index[0]
    
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(sim_matrix[idx]))
    
    # Sort the movies based on the similarity scores (descending)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 2 most similar movies (excluding itself)
    sim_scores = sim_scores[1:3]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    
    # Print results
    print(f"Recommendations for '{title}':")
    for i, idx in enumerate(movie_indices):
        print(f"- {df['movie_title'].iloc[idx]} (Similarity Score: {scores[i]:.2f})")

# Let's test it!
recommend('The Matrix', df, cosine_sim)
