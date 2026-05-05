"""
Example 01: Tokenization
This script demonstrates basic tokenization using built-in Python and NLTK.
"""

import nltk
# Download required NLTK data files (only needs to be run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import word_tokenize, sent_tokenize

text = "Hello there! Welcome to the NLP module. It's going to be fun."

print("--- 1. Python Built-in Split ---")
# Simple split by whitespace
basic_split = text.split()
print(f"Basic Split: {basic_split}")
# Notice how "there!" keeps the exclamation mark attached.

print("\n--- 2. NLTK Word Tokenize ---")
# NLTK is smarter, it separates punctuation from words
nltk_tokens = word_tokenize(text)
print(f"NLTK Tokens: {nltk_tokens}")
# Notice "It's" is split into "It" and "'s".

print("\n--- 3. NLTK Sentence Tokenize ---")
# Splitting by sentence
sentences = sent_tokenize(text)
for i, s in enumerate(sentences):
    print(f"Sentence {i+1}: {s}")
