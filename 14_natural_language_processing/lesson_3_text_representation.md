# Text Representation

## Learning Objective
Students will understand how to convert cleaned, preprocessed text into numerical vectors so that machine learning algorithms can process them. They will learn Bag of Words (BoW), N-grams, and TF-IDF.

## What Is This Topic?
Text representation (also called Feature Extraction or Vectorization) is the bridge between human language and mathematical models. It's the process of turning a list of words into a row of numbers.

## Why This Topic Matters
Machine learning models (like Logistic Regression or Random Forests) expect numerical matrices as input `X`. They cannot do math on the string `"I love this product"`. We must translate text into a numerical format.

## Core Intuition
Imagine a giant spreadsheet. 
- Every **row** is a document (a single review, an email, a tweet).
- Every **column** is a unique word from our entire dataset (our "vocabulary").
If a document contains a word, we put a number in that column. 
- **Bag of Words:** We just put the *count* of how many times the word appeared.
- **TF-IDF:** We put a *weighted score* indicating how important that word is to that specific document.

## Key Concepts

### 1. Document-Term Matrix (DTM)
The final matrix where rows are documents and columns are terms (words). This matrix is usually huge and sparse (mostly zeros, because most documents only use a tiny fraction of the total vocabulary).

### 2. Bag of Words (BoW) / Count Vectorization
- Creates a vocabulary of all unique words.
- For each document, counts the frequency of each word.
- **Problem:** It gives too much weight to frequent words (even if stopwords were removed, words like "good" might dominate without being very informative). It also loses all word order.

### 3. N-grams
Instead of just single words (unigrams), we consider pairs of words (bigrams) or triplets (trigrams).
- **Unigram:** `["not", "good"]`
- **Bigram:** `["not good"]`
- **Why?** It preserves *some* context. "not good" has a completely different meaning than "not" and "good" separately.

### 4. TF-IDF (Term Frequency-Inverse Document Frequency)
A statistical measure used to evaluate how important a word is to a document in a collection or corpus.
- **TF (Term Frequency):** How often does the word appear in *this specific document*? (Higher is better).
- **IDF (Inverse Document Frequency):** How rare is the word across *all documents*? (Rarer is better).
- **Formula:** `TF * IDF`. 
- **Result:** Words that are frequent in one document but rare globally get the highest scores (e.g., "brakes" in a car review).

## Real-World Uses
- Creating the `X` matrix before training a Spam Classifier.
- Finding similar documents (calculating cosine similarity between two TF-IDF vectors to recommend articles).

## Advantages
- Very easy to understand and implement.
- Works surprisingly well for basic classification tasks (like sentiment analysis).

## Limitations
- **Sparsity:** The matrices are huge and take up a lot of memory.
- **Loss of sequence:** "Dog bites man" and "Man bites dog" have the exact same Bag of Words representation.
- **No semantic meaning:** The model doesn't know that "car" and "automobile" are related; they are just two independent columns.

## Related Methods
- Word Embeddings (Word2Vec, GloVe) - these solve the semantic meaning and sparsity problems.

## Code References
- `code/example-01-bag-of-words.py` — Understanding `CountVectorizer`.
- `code/example-02-tfidf.py` — Understanding `TfidfVectorizer`.
- `code/example-03-n-grams.py` — How to use bigrams to capture context.
- `code/example-04-real-world.py` — Applying vectorization to a dataset before training a model.


---

## Text Representation Methods and Options

We primarily use `scikit-learn` for classical text representation.

### `sklearn.feature_extraction.text.CountVectorizer`
Converts a collection of text documents to a matrix of token counts (Bag of Words).

#### Syntax
`vectorizer = CountVectorizer(stop_words='english', max_features=1000)`

#### Common Parameters
- `stop_words='english'`: Automatically removes English stopwords (though doing it manually in preprocessing is often safer).
- `lowercase=True`: Defaults to True. Automatically lowercases text.
- `max_features`: Int. Limits the vocabulary to the top `N` most frequent words. Crucial for memory management!
- `ngram_range`: Tuple `(min_n, max_n)`. e.g., `(1, 2)` means unigrams and bigrams.
- `min_df`: Float or Int. Minimum document frequency. Ignore terms that have a document frequency strictly lower than this threshold (removes extremely rare words / typos).
- `max_df`: Float or Int. Maximum document frequency. Ignore terms that appear in more than this % of documents (removes corpus-specific stopwords).

#### Common Methods
- `fit(X)`: Learns the vocabulary dictionary of all tokens in the raw documents.
- `transform(X)`: Transforms documents to document-term matrix.
- `fit_transform(X)`: Does both in one step. (Use on training data!).
- `get_feature_names_out()`: Returns the list of words that correspond to the columns of the matrix.

### `sklearn.feature_extraction.text.TfidfVectorizer`
Convert a collection of raw documents to a matrix of TF-IDF features.

#### Syntax
`vectorizer = TfidfVectorizer(max_features=1000)`

#### Common Parameters
It inherits almost all parameters from `CountVectorizer` (`max_features`, `ngram_range`, `min_df`, `max_df`).
- `norm`: 'l1', 'l2', or None. Defaults to 'l2' (Normalizes the vectors to have length 1, making cosine similarity easier).

#### Typical Workflow
1. Split data into Train and Test sets.
2. Initialize Vectorizer.
3. `X_train_vec = vectorizer.fit_transform(X_train)` (Learn vocab AND transform).
4. `X_test_vec = vectorizer.transform(X_test)` (ONLY transform based on train vocab. NEVER fit on test data!).

### Understanding the Output (Sparse Matrices)
The output of `fit_transform` is a **SciPy Sparse Matrix**, not a Pandas DataFrame or NumPy array. 
Because 99% of the matrix is zeros, SciPy only stores the locations of the non-zero values to save RAM.
- Use `.toarray()` to convert it to a NumPy array (only do this for small datasets or viewing purposes, otherwise your RAM will crash!).

---

## Examples for Text Representation

This directory contains Python scripts demonstrating how to convert text into numerical matrices.

### Code References
- `code/example-01-bag-of-words.py` — Demonstrates `CountVectorizer` and how to interpret the Document-Term Matrix.
- `code/example-02-tfidf.py` — Demonstrates `TfidfVectorizer` and explains how TF-IDF scores differ from raw counts.
- `code/example-03-n-grams.py` — Shows how altering the `ngram_range` changes the features generated.
- `code/example-04-real-world.py` — Best practices: fitting on train data, transforming on test data, and handling `min_df`/`max_df` to optimize the vocabulary.

---

## Practice: Text Representation

### Concept Questions
1. If you have 1,000 documents and a total unique vocabulary of 5,000 words, what are the dimensions of your Document-Term Matrix?
2. Why is the Document-Term Matrix mostly filled with zeros?
3. Explain why the word "the" would get a very low TF-IDF score even if it appears 100 times in a document.
4. Why must we call `fit_transform` on the training data, but ONLY `transform` on the testing data?

### Coding Tasks
1. **Bag of Words:** Given a list `corpus = ["Machine learning is fun", "Python is great for machine learning"]`, use `CountVectorizer` to create a matrix. Print the feature names (vocabulary) and the matrix as an array.
2. **TF-IDF:** Apply `TfidfVectorizer` to the same corpus. Look at the score for the word "machine" in both sentences. Is it the same? Why or why not?
3. **N-Grams:** Modify your `CountVectorizer` to use an `ngram_range` of `(1, 2)`. Print the new feature names. Notice how "machine learning" is now a single feature.

### Interpretation Tasks
A student limits their vocabulary by setting `max_features=10`. They notice their model performance drops significantly. Why might this happen? Conversely, if they set `max_features=None` on a 100,000 document dataset, what technical problem are they likely to encounter?

---

## Interview Questions: Text Representation

### Beginner Level
1. What is a Document-Term Matrix?
2. Can you explain how Bag of Words works?

### Intermediate Level
3. What does TF-IDF stand for? How does it improve upon Bag of Words?
4. What is an N-gram? Why would you use bigrams in a sentiment analysis task?
5. How does a Sparse Matrix differ from a dense NumPy array, and why do we use them in NLP?

### Advanced / Practical Level
6. You have trained a spam filter using a TF-IDF vectorizer. Tomorrow, an email arrives with a completely new word the model has never seen during training (e.g., "cryptocoinzzz"). How does the vectorizer handle this word? 
   *(Answer: It ignores it. The word is not in the learned vocabulary dictionary, so it drops the feature. This highlights the limitation of fixed vocabularies).*
7. Explain the parameters `min_df` and `max_df` in `scikit-learn` vectorizers. Why are they useful?
8. What is the fundamental limitation of both BoW and TF-IDF when it comes to understanding language semantics? 
   *(Answer: They ignore sequence/grammar, and they don't understand that words with similar meanings (e.g., "happy", "joyful") are related—they are just orthogonal vectors).*

---

## Python Code Examples

### `example-01-bag-of-words.py`

```python
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
```

### `example-02-tfidf.py`

```python
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
```

### `example-03-n-grams.py`

```python
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
```

### `example-04-real-world.py`

```python
"""
Example 04: Real World Vectorization Workflow
Demonstrates Train/Test splits, min_df, max_df, and max_features.
"""

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample dataset
documents = [
    "Data science is the sexiest job.",
    "Python is great for data science.",
    "I love writing Python code.",
    "Machine learning models are powerful.",
    "Data data data everywhere.",
    "Deep learning is a subset of machine learning."
]
labels = [1, 1, 1, 0, 1, 0] # Fake labels

print("--- 1. Train Test Split ---")
# ALWAYS split before vectorizing!
X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.33, random_state=42)
print(f"Training docs: {len(X_train)}")
print(f"Testing docs: {len(X_test)}\n")

print("--- 2. Initialize Vectorizer with Constraints ---")
vectorizer = TfidfVectorizer(
    stop_words='english', # Remove stopwords
    max_features=10,      # Keep only top 10 words (prevents huge memory usage in real apps)
    min_df=2,             # Word must appear in at least 2 documents
    max_df=0.9            # Word must not appear in more than 90% of documents
)

print("--- 3. Fit and Transform on TRAIN ---")
# The vectorizer learns the vocabulary ONLY from the training data
X_train_vec = vectorizer.fit_transform(X_train)
print(f"Learned Vocabulary: {vectorizer.get_feature_names_out()}")
print(f"Train Matrix Shape: {X_train_vec.shape}")

print("\n--- 4. ONLY Transform on TEST ---")
# The vectorizer uses the already learned vocabulary to score the test data
# NEVER call .fit() or .fit_transform() on X_test!
X_test_vec = vectorizer.transform(X_test)
print(f"Test Matrix Shape: {X_test_vec.shape}")

print("\nCrucial Concept:")
print("If a word exists in X_test but was not seen in X_train (or didn't meet min_df), it is completely ignored. This simulates real life where a deployed model encounters words it has never seen before.")
```
