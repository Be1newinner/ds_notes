import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example 03: Building a User Profile Vector
# Instead of recommending based on a single movie, we build an "average" profile
# of what the user likes based on their history.

print("--- User Profile Content Recommender ---")

# 1. Dataset of articles
articles = pd.DataFrame({
    'article_id': [0, 1, 2, 3, 4],
    'title': ['AI in Healthcare', 'Stock Market drops', 'Machine Learning trends', 'Sports results', 'Deep Learning for Medical Imaging'],
    'text': [
        'AI is revolutionizing healthcare and hospitals.',
        'The stock market saw a massive drop today.',
        'Machine learning trends show growth in AI tech.',
        'Local sports team wins the championship.',
        'Deep learning models are reading medical x-rays.'
    ]
})

# 2. Vectorize Articles
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(articles['text'])

# 3. Simulate a User's History
# Let's say User A read and liked article 0 and article 2.
user_history_indices = [0, 2] 

print(f"User liked: '{articles['title'].iloc[0]}' and '{articles['title'].iloc[2]}'")

# 4. Build the User Profile Vector
# We grab the mathematical vectors for the articles they liked
liked_vectors = tfidf_matrix[user_history_indices]

# We average them together to create a single vector representing their taste
# (We use mean across axis=0 to average the columns)
user_profile_vector = np.asarray(liked_vectors.mean(axis=0))

# 5. Find recommendations
# We calculate the cosine similarity between the NEW user profile vector and ALL articles
sim_scores = cosine_similarity(user_profile_vector, tfidf_matrix)

# 6. Rank the articles
# sim_scores is a 2D array, we flatten it to 1D
sim_scores_1d = sim_scores.flatten()

# Get indices sorted by score, descending
sorted_indices = sim_scores_1d.argsort()[::-1]

print("\nTop Recommendations for User Profile:")
for idx in sorted_indices:
    # Don't recommend articles they already read
    if idx not in user_history_indices:
        print(f"- {articles['title'].iloc[idx]} (Score: {sim_scores_1d[idx]:.2f})")
