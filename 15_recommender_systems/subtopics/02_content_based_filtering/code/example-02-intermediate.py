import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Example 02: Text-Based Recommender (TF-IDF)
# Using Natural Language plot descriptions to recommend items.

print("--- Text-Based Content Recommender (TF-IDF) ---")

# 1. Mock dataset of movies and their Plot Descriptions
data = {
    'movie_title': ['The Dark Knight', 'Batman Begins', 'The Avengers', 'Superman', 'A Beautiful Mind'],
    'plot': [
        'Batman fights the Joker in Gotham City to save the people.',
        'Bruce Wayne trains to become Batman and saves Gotham City from destruction.',
        'Superheroes team up to save the world from an alien invasion.',
        'An alien orphan becomes a superhero to save Earth.',
        'A brilliant mathematician struggles with schizophrenia.'
    ]
}
df = pd.DataFrame(data)

# 2. Vectorize the text using TF-IDF
# We use stop_words='english' to remove 'the', 'a', 'to', etc.
tfidf = TfidfVectorizer(stop_words='english')

# fit_transform learns the vocabulary and transforms the text into vectors
tfidf_matrix = tfidf.fit_transform(df['plot'])

print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
# Shows (5 movies, X unique words)

# 3. Calculate Cosine Similarity
# Because TF-IDF vectors are already normalized by sklearn, we can use linear_kernel
# which is mathematically equivalent to cosine_similarity but faster.
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# 4. Recommendation Function (Using Pandas indexing)
# Create a reverse mapping of indices and movie titles
indices = pd.Series(df.index, index=df['movie_title']).drop_duplicates()

def recommend_text(title, df, sim_matrix):
    idx = indices[title]
    
    # Get pairwise similarity scores
    sim_scores = list(enumerate(sim_matrix[idx]))
    
    # Sort them
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top 2
    sim_scores = sim_scores[1:3]
    
    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    
    print(f"\nRecommendations for '{title}':")
    for i, m_idx in enumerate(movie_indices):
         print(f"- {df['movie_title'].iloc[m_idx]} (Sim: {scores[i]:.2f})")

# Let's see what happens if we like Batman
recommend_text('The Dark Knight', df, cosine_sim)

# Let's see what happens if we like The Avengers
recommend_text('The Avengers', df, cosine_sim)
