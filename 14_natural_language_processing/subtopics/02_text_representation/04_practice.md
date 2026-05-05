# Practice: Text Representation

## Concept Questions
1. If you have 1,000 documents and a total unique vocabulary of 5,000 words, what are the dimensions of your Document-Term Matrix?
2. Why is the Document-Term Matrix mostly filled with zeros?
3. Explain why the word "the" would get a very low TF-IDF score even if it appears 100 times in a document.
4. Why must we call `fit_transform` on the training data, but ONLY `transform` on the testing data?

## Coding Tasks
1. **Bag of Words:** Given a list `corpus = ["Machine learning is fun", "Python is great for machine learning"]`, use `CountVectorizer` to create a matrix. Print the feature names (vocabulary) and the matrix as an array.
2. **TF-IDF:** Apply `TfidfVectorizer` to the same corpus. Look at the score for the word "machine" in both sentences. Is it the same? Why or why not?
3. **N-Grams:** Modify your `CountVectorizer` to use an `ngram_range` of `(1, 2)`. Print the new feature names. Notice how "machine learning" is now a single feature.

## Interpretation Tasks
A student limits their vocabulary by setting `max_features=10`. They notice their model performance drops significantly. Why might this happen? Conversely, if they set `max_features=None` on a 100,000 document dataset, what technical problem are they likely to encounter?
