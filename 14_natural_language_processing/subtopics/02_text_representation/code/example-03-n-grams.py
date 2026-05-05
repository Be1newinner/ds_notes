"""
Example 03: N-Grams
Capturing context by grouping consecutive words.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "This movie is not good.",
    "This movie is good."
]

print("--- 1. Default (Unigrams Only) ---")
vec_unigram = CountVectorizer(ngram_range=(1, 1))
df_uni = pd.DataFrame(vec_unigram.fit_transform(corpus).toarray(), 
                      columns=vec_unigram.get_feature_names_out())
print(df_uni)
print("Observation: Both documents have a '1' for 'good'. The model doesn't know the first doc is negative!")

print("\n--- 2. Bigrams Only ---")
vec_bigram = CountVectorizer(ngram_range=(2, 2))
df_bi = pd.DataFrame(vec_bigram.fit_transform(corpus).toarray(), 
                     columns=vec_bigram.get_feature_names_out())
print(df_bi)
print("Observation: Now we have 'not good' vs 'is good'. Context is captured!")

print("\n--- 3. Unigrams AND Bigrams ---")
# Usually, we use a mix of both (1, 2)
vec_mix = CountVectorizer(ngram_range=(1, 2))
df_mix = pd.DataFrame(vec_mix.fit_transform(corpus).toarray(), 
                      columns=vec_mix.get_feature_names_out())
print(df_mix)
print(f"Total features created: {len(vec_mix.get_feature_names_out())}")
print("Warning: Adding n-grams drastically increases the number of columns (features) in your dataset!")
