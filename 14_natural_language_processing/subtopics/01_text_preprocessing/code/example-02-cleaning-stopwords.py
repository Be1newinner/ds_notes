"""
Example 02: Cleaning and Stopwords
This script demonstrates lowercasing, removing punctuation, and filtering stopwords.
"""

import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

text = "The quick brown fox, jumps over the lazy, sleeping dog! Isn't that crazy?"

# 1. Lowercase the text
text_lower = text.lower()
print(f"Lowercased: {text_lower}\n")

# 2. Tokenize
tokens = word_tokenize(text_lower)
print(f"Tokens: {tokens}\n")

# 3. Remove Punctuation
# We keep tokens that have alphabetic characters
tokens_no_punct = [word for word in tokens if word.isalpha()]
print(f"No Punctuation: {tokens_no_punct}\n")

# 4. Remove Stopwords
stop_words = set(stopwords.words('english'))
clean_tokens = [word for word in tokens_no_punct if word not in stop_words]

print(f"Final Clean Tokens (No Stopwords): {clean_tokens}")
print(f"Notice words like 'the', 'over', 'that' are gone.")
