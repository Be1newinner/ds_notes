"""
Example 02: TF-IDF (Term Frequency - Inverse Document Frequency)
Converts text to a matrix of weighted scores.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "The quick brown fox.", # Doc 1
    "The slow brown bear.", # Doc 2
    "The quick quick fox."  # Doc 3
]

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform
X_tfidf = vectorizer.fit_transform(corpus)

# Get feature names
vocab = vectorizer.get_feature_names_out()

# Create DataFrame for visualization
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=vocab, index=["Doc 1", "Doc 2", "Doc 3"])

print("--- TF-IDF Document-Term Matrix ---")
# Rounding to 2 decimal places for readability
print(df_tfidf.round(2))

print("\n--- Analysis ---")
print("1. Look at the word 'the'. It appears in every document. Because it's so common across the corpus (low IDF), its TF-IDF score is relatively low (~0.43 to ~0.35) even though it appears 1 time in every document.")
print("2. Look at 'bear' in Doc 2. It only appears in Doc 2 (high IDF). Therefore, it gets a much higher score (0.68) than 'the' or 'brown' in that same document.")
print("3. Look at 'quick' in Doc 3. It appears twice (high TF). Combined with the fact that it's not in every document, it gets the highest score (0.83).")
