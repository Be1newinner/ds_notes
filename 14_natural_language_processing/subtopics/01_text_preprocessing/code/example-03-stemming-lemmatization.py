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
