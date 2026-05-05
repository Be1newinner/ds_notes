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
