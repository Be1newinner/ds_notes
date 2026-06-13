# Text Preprocessing

## Learning Objective
Students should understand what text preprocessing is, why it is essential for NLP tasks, and the specific steps involved (tokenization, lowercasing, removing punctuation/stopwords, stemming, and lemmatization).

## What Is This Topic?
Text preprocessing is the process of cleaning and preparing unstructured text data so that it can be converted into numbers and fed into a machine learning model. It's like data cleaning but specifically for words.

## Why This Topic Matters
Raw text is incredibly messy. "Apple", "apple", "APPLE!", and "apples" might all mean the same thing, but a computer sees them as four completely different entities. If we don't clean the text, our machine learning models will have too many features (words) and won't be able to learn general patterns.

## Core Intuition
Imagine trying to read a book where every single word, punctuation mark, and capitalization is treated as a unique symbol. It would be overwhelming. Text preprocessing simplifies the text. We break it into chunks (tokens), make everything lowercase so "Hello" == "hello", remove useless words like "the" or "and" (stopwords), and reduce words to their base form so "running" == "run" (stemming/lemmatization).

## Key Concepts
- **Tokenization:** Breaking text into smaller units (usually words or sentences).
- **Lowercasing:** Converting all characters to lowercase.
- **Punctuation Removal:** Stripping out commas, periods, etc.
- **Stopwords:** Common words that add little meaning (e.g., "is", "at", "which").
- **Stemming:** Cutting off the ends of words to find the root (often results in non-words like "comput").
- **Lemmatization:** Using a dictionary to find the actual linguistic root of a word (e.g., "better" -> "good").

## Step-by-Step Explanation
1.  **Read the raw text:** Get the string of data.
2.  **Lowercase:** Convert everything to lowercase.
3.  **Tokenize:** Split the string into a list of words.
4.  **Remove Punctuation & Numbers:** Filter out non-alphabetic tokens.
5.  **Remove Stopwords:** Filter out tokens that are in a predefined list of stopwords.
6.  **Stem/Lemmatize:** Apply a function to reduce remaining words to their base form.
7.  **Rejoin (Optional):** Sometimes you join the tokens back into a single clean string for the next step.

## Important Parameters / Options / Settings
- **Language:** Preprocessing depends entirely on the language. Stopwords in English are different from Spanish.
- **Aggressiveness:** Stemming is aggressive and fast; Lemmatization is accurate but slow.

## Output / Result Interpretation
The output of preprocessing is either a list of clean tokens `['quick', 'brown', 'fox', 'jump']` or a single clean string `"quick brown fox jump"`.

## Real-World Uses
- Pre-step for Search Engines (so searching "running shoes" finds "run shoe").
- Pre-step for Sentiment Analysis (removing noise so only emotion-carrying words matter).

## Advantages
- Greatly reduces the size of the vocabulary.
- Helps models generalize better by grouping similar words.

## Limitations
- You can accidentally remove important context (e.g., "not good" might lose "not" if "not" is a stopword).
- Stemming can create meaningless words, making debugging harder.

## Common Mistakes
- Removing punctuation *before* dealing with contractions (e.g., "don't" becomes "dont", then just "don" and "t").
- Blindly removing stopwords without considering the task. In sentiment analysis, "not" is a crucial word!

## Related Methods
- POS Tagging (Part of Speech) - identifying if a word is a noun, verb, etc., which helps lemmatization.

## Code References
- `code/example-01-tokenization.py` — simple tokenization with NLTK
- `code/example-02-cleaning-stopwords.py` — lowercasing, punctuation, and stopwords
- `code/example-03-stemming-lemmatization.py` — reducing words to root forms
- `code/example-04-real-world.py` — full pipeline on a sample dataset


---

## Text Preprocessing Methods and Options

### NLTK (Natural Language Toolkit)
The standard library for basic NLP in Python.

#### `nltk.tokenize.word_tokenize(text)`
- **Purpose:** Splits text into words and punctuation tokens.
- **Syntax:** `tokens = word_tokenize("Hello, world!")`
- **Return Type:** List of strings.
- **Workflow:** Usually the first step after lowercasing.

#### `nltk.corpus.stopwords.words('english')`
- **Purpose:** Returns a list of common English stopwords.
- **Return Type:** List of strings.
- **Common Usage:** Used in a list comprehension to filter tokens: `[w for w in tokens if w not in stop_words]`

#### `nltk.stem.PorterStemmer`
- **Purpose:** Aggressively chops off word endings.
- **Syntax:** `stemmer = PorterStemmer(); stemmer.stem("running")`
- **Return Type:** String.

#### `nltk.stem.WordNetLemmatizer`
- **Purpose:** Linguistically maps words to their dictionary root. Requires POS tags for accuracy.
- **Syntax:** `lemmatizer = WordNetLemmatizer(); lemmatizer.lemmatize("better", pos="a")`
- **Return Type:** String.

### SpaCy
A modern, fast, production-ready NLP library.

#### `spacy.load('en_core_web_sm')`
- **Purpose:** Loads a pre-trained language model.
- **Syntax:** `nlp = spacy.load("en_core_web_sm"); doc = nlp("This is a sentence.")`
- **Workflow:** SpaCy processes the entire document at once, automatically doing tokenization, POS tagging, and lemmatization.
- **Attributes on Tokens:**
  - `token.text`: The raw text.
  - `token.is_stop`: Boolean indicating if it's a stopword.
  - `token.is_punct`: Boolean indicating if it's punctuation.
  - `token.lemma_`: The lemmatized base form of the word.

### Common Mistakes & Best Practices
- **Best Practice:** When doing simple tasks like TF-IDF, NLTK stemming is often enough. For complex tasks needing grammatical context, use SpaCy lemmatization.
- **Mistake:** Forgetting to download NLTK data (`nltk.download('punkt')`) before running scripts.

---

## Examples for Text Preprocessing

This directory contains Python scripts demonstrating how to clean and prepare text data.

### Code References
- `code/example-01-tokenization.py` — Introduction to splitting text into words and sentences using NLTK and basic Python.
- `code/example-02-cleaning-stopwords.py` — Shows how to lowercase text, remove punctuation, and filter out common stopwords.
- `code/example-03-stemming-lemmatization.py` — Compares the aggressive chopping of Stemming with the dictionary-based approach of Lemmatization.
- `code/example-04-real-world.py` — Combines all steps into a single preprocessing pipeline function and applies it to a pandas DataFrame containing sample reviews.

---

## Practice: Text Preprocessing

### Concept Questions
1. Why is lowercasing generally the very first step in text preprocessing?
2. What is the difference between a tokenizer that splits by space (`text.split()`) and `nltk.word_tokenize()`?
3. Give an example of a situation where removing stopwords would be a **bad** idea.
4. Explain the difference between Stemming and Lemmatization to a non-technical person.

### Coding Tasks
1. **Basic Clean:** Write a Python function using base Python (no NLTK/SpaCy) that takes a string, makes it lowercase, removes commas and periods, and splits it by spaces.
2. **Stopword Filter:** Given the text `"The quick brown fox jumps over the lazy dog"`, use NLTK to tokenize it and remove all English stopwords. Print the resulting list.
3. **Stem vs Lemma Challenge:** Run the word `"organization"` through NLTK's `PorterStemmer` and `WordNetLemmatizer`. What are the outputs? Why are they different?

### Interpretation Tasks
Look at this output from a stemmer:
`['appl', 'comput', 'are', 'veri', 'expens']`
What do you think the original sentence was? What problems might this aggressive stemming cause if we try to read the data manually later?

---

## Interview Questions: Text Preprocessing

### Beginner Level
1. What are stopwords, and why do we usually remove them in NLP?
2. Can you explain tokenization?

### Intermediate Level
3. What is the difference between stemming and lemmatization? When would you choose one over the other?
4. How would you handle punctuation in text data? Are there cases where you would keep it? (Hint: Emoticons like ":)" or exclamation marks in sentiment analysis).

### Advanced / Practical Level
5. You are building an NLP model to classify legal documents. Would you use stemming or lemmatization? Why?
6. If you notice your model is misclassifying sentences that contain negative sentiment (e.g., "The product was not good"), what preprocessing step might be causing the issue, and how would you fix it?

### Output Interpretation
7. If a lemmatizer turns "leaves" into "leaf", but you wanted it to mean "departing" (as in "he leaves the room"), what missing information does the lemmatizer need to get it right? (Answer: Part of Speech tag).

---

## Python Code Examples

### `example-01-tokenization.py`

```python
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
```

### `example-02-cleaning-stopwords.py`

```python
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
```

### `example-03-stemming-lemmatization.py`

```python
"""
Example 03: Stemming vs. Lemmatization
Comparing aggressive truncation (stemming) with dictionary-based roots (lemmatization).
"""

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

words_to_process = ["running", "runs", "ran", "better", "organizations", "caring"]

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print(f"{'Original':<15} | {'Stemmed':<15} | {'Lemmatized (Verb)':<15}")
print("-" * 50)

for word in words_to_process:
    stemmed = stemmer.stem(word)
    # Lemmatizer needs to know the Part of Speech (POS). We default to Verb ('v') here for demonstration
    lemmatized = lemmatizer.lemmatize(word, pos='v') 
    
    # Note for 'better', 'v' is wrong (it's an adjective), so let's check adjective ('a') as well
    if word == "better":
         lemmatized = lemmatizer.lemmatize(word, pos='a')
         
    print(f"{word:<15} | {stemmed:<15} | {lemmatized:<15}")

print("\nObservation:")
print("- Stemming chops 'organizations' to 'organ', which isn't a real word.")
print("- Lemmatization turns 'ran' to 'run', understanding the past tense.")
print("- Stemming turns 'caring' to 'care' (or 'car' sometimes), Lemmatization correctly keeps 'care'.")
```

### `example-04-real-world.py`

```python
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
```
