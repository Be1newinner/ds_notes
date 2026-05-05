"""
Example 01: Bag of Words (CountVectorizer)
Converts text into a matrix of token counts.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Sample corpus (collection of documents)
corpus = [
    "I love dogs.",
    "I love cats too.",
    "Dogs are great, but cats are cats."
]

print("--- 1. Initializing CountVectorizer ---")
# By default, it lowercases and removes punctuation
vectorizer = CountVectorizer()

print("\n--- 2. Fitting and Transforming ---")
# Learn the vocabulary and create the matrix
X = vectorizer.fit_transform(corpus)

# X is a sparse matrix. We convert it to a dense array just to print it easily
X_array = X.toarray()

print("\n--- 3. Understanding the Output ---")
# Get the vocabulary
vocab = vectorizer.get_feature_names_out()
print(f"Vocabulary (Columns):\n{vocab}")

print(f"\nMatrix (Rows=Documents, Columns=Words):\n{X_array}")

# Let's make it beautiful with Pandas
df_dtm = pd.DataFrame(X_array, columns=vocab, index=["Doc 1", "Doc 2", "Doc 3"])
print("\n--- 4. Document-Term Matrix (DTM) ---")
print(df_dtm)

print("\nNotice how Doc 3 has a '2' for 'are' and 'cats' because they appear twice.")
