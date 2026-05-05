# Introduction to Word Embeddings

## Learning Objective
Students should understand the limitations of Bag of Words/TF-IDF and learn how Word Embeddings (like Word2Vec) solve these problems by representing words as dense vectors capturing semantic meaning.

## What Is This Topic?
Word Embeddings are a modern way to represent text. Instead of a sparse matrix where every column is a unique word (and most values are zero), we represent each word as a "dense" array of numbers (e.g., 300 numbers) where the numbers encode the *meaning* of the word.

## Why This Topic Matters
This is the foundation of all modern NLP (including ChatGPT). Classical NLP (TF-IDF) thinks "cat" and "dog" are entirely unrelated concepts. Word embeddings know that "cat" and "dog" are both pets and map them very close to each other in mathematical space.

## Core Intuition
Imagine plotting words on a 2D graph. The X-axis is "How much is this a living thing?" and the Y-axis is "How much is this associated with royalty?".
- "King" = [0.9, 0.9]
- "Queen" = [0.9, 0.9] (They are close together!)
- "Apple" = [0.2, 0.0] (Far away)
- "Car" = [0.0, 0.0]
In reality, word embeddings have 100 to 300 dimensions, representing incredibly complex relationships learned by reading millions of Wikipedia articles.

## Key Concepts

### 1. Dense vs. Sparse Vectors
- **TF-IDF (Sparse):** Length is 10,000 (vocab size). mostly 0s. 1 means the word is there.
- **Embeddings (Dense):** Length is 300. Every number is a non-zero float. It represents the *concept* of the word.

### 2. Semantic Similarity
Because words are coordinates in space, we can use math (Cosine Similarity) to find synonyms. `similarity("happy", "joyful")` will be very high (near 1.0).

### 3. Word2Vec (The Algorithm)
Developed by Google in 2013. It learns these embeddings by training a shallow neural network to predict a word based on its surrounding words (or vice versa). "You shall know a word by the company it keeps."

### 4. Vector Arithmetic (The Magic)
Because meaning is encoded geometrically, you can do math with words!
`Vector("King") - Vector("Man") + Vector("Woman") ≈ Vector("Queen")`

## Real-World Uses
- Improving Search Engines (searching "affordable laptop" brings up results for "cheap computer" because the vectors are similar).
- Recommendation Systems (articles with similar meaning).
- Input layer for Deep Learning NLP models (LSTMs, Transformers).

## Advantages
- Captures deep semantic meaning and relationships (synonyms, analogies).
- Reduces dimensionality drastically (from 50,000 to 300).
- Models generalize better to words they haven't seen in the exact training context.

## Limitations
- Static embeddings (like Word2Vec) can't handle context well. The word "bank" has the same vector whether it's a "river bank" or a "bank account". (Modern Transformers solve this).
- They require massive amounts of text and computing power to train from scratch (which is why we almost always use pre-trained embeddings).

## Code References
- `code/example-01-word2vec-gensim.py` — Training your own Word2Vec model on custom text using `gensim`.
- `code/example-02-spacy-vectors.py` — Using powerful pre-trained embeddings provided by `spacy`.
- `code/example-03-real-world.py` — Averaging word vectors to classify a whole sentence.
