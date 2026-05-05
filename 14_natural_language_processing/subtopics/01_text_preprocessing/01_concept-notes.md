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
