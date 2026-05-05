"""
Example 01: Training Word2Vec with Gensim
Learning embeddings from scratch on a custom corpus.
"""

# pip install gensim
from gensim.models import Word2Vec

# 1. Dataset (Must be a list of tokenized sentences)
sentences = [
    ["i", "love", "coding", "in", "python"],
    ["python", "is", "a", "great", "programming", "language"],
    ["i", "enjoy", "writing", "code", "in", "python"],
    ["java", "is", "another", "programming", "language"],
    ["coding", "in", "java", "is", "fun"]
]

print("--- 1. Training the Model ---")
# vector_size: Dimensions of the embedding (usually 100-300, we use 10 for this tiny example)
# window: How many surrounding words to look at
# min_count: Ignore words that appear less than this
model = Word2Vec(sentences=sentences, vector_size=10, window=2, min_count=1)
print("Model trained successfully.")

print("\n--- 2. Accessing Vectors ---")
vector_python = model.wv["python"]
print(f"Vector for 'python' (Length: {len(vector_python)}):\n{vector_python}")

print("\n--- 3. Finding Similar Words ---")
# The model learns that 'python' and 'java' appear in similar contexts (near 'programming', 'language', 'in')
similar_to_python = model.wv.most_similar("python", topn=3)
print("Words most similar to 'python':")
for word, score in similar_to_python:
    print(f" - {word}: {score:.4f}")

similar_to_coding = model.wv.most_similar("coding", topn=2)
print("\nWords most similar to 'coding':")
for word, score in similar_to_coding:
    print(f" - {word}: {score:.4f}")
    
print("\nNote: Because our training data is incredibly small, the similarities might not be perfect. Real Word2Vec requires millions of words.")
