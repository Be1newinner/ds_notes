# Interview Questions: Text Representation

## Beginner Level
1. What is a Document-Term Matrix?
2. Can you explain how Bag of Words works?

## Intermediate Level
3. What does TF-IDF stand for? How does it improve upon Bag of Words?
4. What is an N-gram? Why would you use bigrams in a sentiment analysis task?
5. How does a Sparse Matrix differ from a dense NumPy array, and why do we use them in NLP?

## Advanced / Practical Level
6. You have trained a spam filter using a TF-IDF vectorizer. Tomorrow, an email arrives with a completely new word the model has never seen during training (e.g., "cryptocoinzzz"). How does the vectorizer handle this word? 
   *(Answer: It ignores it. The word is not in the learned vocabulary dictionary, so it drops the feature. This highlights the limitation of fixed vocabularies).*
7. Explain the parameters `min_df` and `max_df` in `scikit-learn` vectorizers. Why are they useful?
8. What is the fundamental limitation of both BoW and TF-IDF when it comes to understanding language semantics? 
   *(Answer: They ignore sequence/grammar, and they don't understand that words with similar meanings (e.g., "happy", "joyful") are related—they are just orthogonal vectors).*
