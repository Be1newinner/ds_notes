"""
Example 04: Real World Preprocessing Pipeline
Combining all steps into a reusable function and applying it to a pandas DataFrame.
"""

import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Sample Dataset
data = {
    'review_id': [1, 2, 3],
    'text': [
        "I absolutely LOVED this product!! It works great.",
        "Terrible experience. The shipping was late and the item is broken.",
        "It's okay, not the best but it gets the job done..."
    ]
}
df = pd.DataFrame(data)

# Initialize tools once
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """A complete preprocessing pipeline."""
    # 1. Lowercase
    text = text.lower()
    
    # 2. Tokenize
    tokens = word_tokenize(text)
    
    # 3. Clean and filter
    clean_tokens = []
    for word in tokens:
        # Keep only alphabetic tokens and remove stopwords
        if word.isalpha() and word not in stop_words:
            # 4. Lemmatize (defaulting to noun, for simplicity in this example)
            lemma = lemmatizer.lemmatize(word)
            clean_tokens.append(lemma)
            
    # 5. Join back into a single string (often required for ML models like TF-IDF)
    return " ".join(clean_tokens)

print("--- Original DataFrame ---")
print(df['text'])

print("\n--- Applying Preprocessing Pipeline ---")
df['clean_text'] = df['text'].apply(preprocess_text)
print(df[['text', 'clean_text']])
