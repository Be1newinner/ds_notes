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


---

## Methods and Options: Word Embeddings

### `gensim.models.Word2Vec`
The most popular library for training your own embeddings from scratch.

#### Syntax
```python
from gensim.models import Word2Vec
model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1)
```

#### Parameters
- `sentences`: A list of lists of tokens (e.g., `[["hello", "world"], ["i", "love", "nlp"]]`).
- `vector_size`: Dimensionality of the word vectors (typically 100-300).
- `window`: Maximum distance between the current and predicted word within a sentence.
- `min_count`: Ignores all words with total frequency lower than this.

#### Common Methods
- `model.wv.most_similar("word")`: Returns a list of words most similar to the given word.
- `model.wv.similarity("word1", "word2")`: Computes the cosine similarity between two words.
- `model.wv["word"]`: Retrieves the actual numpy array (the vector) for the word.

### `spacy` (Pre-trained Models)
The easiest way to use embeddings trained by Google or Facebook on billions of words.

#### Setup
You must download a medium (`md`) or large (`lg`) model. The small (`sm`) model does NOT include real word vectors to save space.
`python -m spacy download en_core_web_md`

#### Syntax and Workflow
```python
import spacy
nlp = spacy.load("en_core_web_md")
doc = nlp("I love apples")

# Get vector for a specific word
apple_vector = doc[2].vector 

# Get similarity between two documents
doc1 = nlp("I like dogs")
doc2 = nlp("I prefer puppies")
print(doc1.similarity(doc2)) 
```

#### Why SpaCy is great
It automatically averages the word vectors in a sentence to give you a single `doc.vector` representing the whole sentence's meaning.

---

## Examples for Word Embeddings

This directory contains Python scripts demonstrating how to generate and use dense word vectors.

### Code References
- `code/example-01-word2vec-gensim.py` — Shows how to train a custom Word2Vec model on a small text corpus and find similar words.
- `code/example-02-spacy-vectors.py` — Demonstrates how to load large, pre-trained word vectors to find semantic similarities between different sentences.
- `code/example-03-real-world.py` — Shows a practical workflow of converting sentences into average word vectors and using them to train a Logistic Regression classifier.

---

## Practice: Word Embeddings

### Concept Questions
1. What is the difference between a sparse vector (TF-IDF) and a dense vector (Word2Vec)?
2. If two words appear in similar contexts (e.g., "I drink coffee" and "I drink tea"), what will happen to their Word2Vec representations during training?
3. Why do we almost always download pre-trained embeddings (like GloVe or SpaCy's models) instead of training our own from scratch?

### Coding Tasks
1. **Gensim Training:** Create a tiny corpus: `[["the", "king", "rules"], ["the", "queen", "rules"]]`. Train a `Word2Vec` model using `gensim` on this data with `vector_size=5`. Print the vector for "king".
2. **SpaCy Similarity:** Load `en_core_web_md` using `spacy`. Calculate and print the similarity between the sentence "I want to buy a car" and "I am looking to purchase an automobile". (Even though they share few words, the similarity should be high).

### Interpretation Tasks
You are building a search engine for a medical database. A doctor searches for "heart attack". The database has a great article titled "Myocardial Infarction Symptoms". 
If you used TF-IDF, would the article show up in the search results? Why or why not? 
If you used Word Embeddings to compare the search query to the article titles, would it show up? Why?

---

## Interview Questions: Word Embeddings

### Beginner Level
1. Can you explain the basic idea behind Word Embeddings to someone without a technical background?
2. What is Cosine Similarity, and why is it used so often with word vectors?

### Intermediate Level
3. How does the Word2Vec algorithm actually learn the representations? What is the neural network trying to predict? *(Answer: It's a fake task! It tries to predict a word given its neighbors (CBOW) or predict neighbors given a word (Skip-Gram). The "hidden weights" learned become the embeddings).*
4. What does the famous equation `King - Man + Woman = Queen` actually mean in the context of vector mathematics?
5. How would you represent an entire sentence using word embeddings so that you can feed it into a standard classifier like Logistic Regression? *(Answer: The simplest way is to take the mean average of all the word vectors in the sentence).*

### Advanced / Practical Level
6. What is the "Out Of Vocabulary" (OOV) problem in Word2Vec, and how do subword models like FastText solve it?
7. Word2Vec produces "static" embeddings. The word "apple" has the same vector whether the sentence is "I ate an apple" or "I bought Apple stock". How do modern architectures like Transformers (BERT) address this limitation? *(Answer: They produce contextualized embeddings, where the vector for "apple" changes dynamically based on the surrounding words in the exact sentence).*

---

## Python Code Examples

### `example-01-word2vec-gensim.py`

```python
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
```

### `example-02-spacy-vectors.py`

```python
"""
Example 02: Pre-trained Embeddings with SpaCy
Using enterprise-grade vectors that already know the English language.

REQUIREMENT: You must download the medium model first in your terminal!
python -m spacy download en_core_web_md
"""

import spacy

print("Loading SpaCy Medium Model (this contains 20,000 word vectors)...")
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("ERROR: Please run `python -m spacy download en_core_web_md` in your terminal first!")
    exit()

print("--- 1. Semantic Similarity (Words) ---")
word1 = nlp("dog")
word2 = nlp("puppy")
word3 = nlp("car")

print(f"Similarity (dog vs puppy): {word1.similarity(word2):.2f}")
print(f"Similarity (dog vs car):   {word1.similarity(word3):.2f}")

print("\n--- 2. Semantic Similarity (Sentences) ---")
# SpaCy automatically averages the word vectors in a sentence to create a document vector.
sent1 = nlp("I want to buy an automobile.")
sent2 = nlp("I am looking to purchase a car.")
sent3 = nlp("I like to eat delicious food.")

print(f"Sent 1: {sent1}")
print(f"Sent 2: {sent2}")
print(f"Sent 3: {sent3}\n")

# Notice that sent1 and sent2 share NO important words (automobile/buy vs car/purchase)
# Yet, the model knows they mean exactly the same thing. TF-IDF would give a similarity of 0!
print(f"Similarity (Sent 1 vs Sent 2): {sent1.similarity(sent2):.2f}  <-- Magic!")
print(f"Similarity (Sent 1 vs Sent 3): {sent1.similarity(sent3):.2f}")

print("\n--- 3. Inspecting the Vector ---")
doc = nlp("AI is the future")
print(f"Shape of the sentence vector: {doc.vector.shape}")
print("This 300-dimension vector can now be passed as 'X' into a Logistic Regression model!")
```

### `example-03-real-world.py`

```python
"""
Example 03: Text Classification with Embeddings
How to use pre-trained word vectors to train a standard scikit-learn classifier.
"""

import spacy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Loading SpaCy Model...")
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("ERROR: Please run `python -m spacy download en_core_web_md` in your terminal first!")
    exit()

# 1. Dataset
sentences = [
    "I absolutely loved the meal, it was delicious.",
    "The food was terrible and cold.",
    "Great service, highly recommend this place.",
    "I hated it, the waiter was rude.",
    "Awesome experience, five stars.",
    "Awful. Do not go here."
]
labels = [1, 0, 1, 0, 1, 0] # 1=Positive, 0=Negative

print("--- 1. Vectorizing with SpaCy ---")
# Instead of TfidfVectorizer, we ask SpaCy for the .vector of each sentence
# We create an empty list, then append the 300-dim numpy array for each sentence
X_vectors = []
for text in sentences:
    doc = nlp(text)
    X_vectors.append(doc.vector)

# Convert list of arrays into a 2D numpy matrix (X)
X = np.array(X_vectors)
y = np.array(labels)

print(f"Feature Matrix Shape: {X.shape} (Documents x 300 dimensions)")

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("\n--- 2. Training the Model ---")
# We just pass the dense arrays into a standard ML algorithm
model = LogisticRegression()
model.fit(X_train, y_train)

# 3. Evaluation
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100}%")

print("\n--- 3. Inference on New Data ---")
new_review = "The pasta was exquisite!"
# We must vectorize the new review the exact same way
new_vec = nlp(new_review).vector

# Sklearn expects a 2D array for prediction: .reshape(1, -1) handles this for a single sample
prediction = model.predict(new_vec.reshape(1, -1))
sentiment = "Positive" if prediction[0] == 1 else "Negative"

print(f"Review: '{new_review}'")
print(f"Prediction: {sentiment}")
print("Notice the model predicted Positive even though it never saw the words 'pasta' or 'exquisite' during training!")
```
