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
