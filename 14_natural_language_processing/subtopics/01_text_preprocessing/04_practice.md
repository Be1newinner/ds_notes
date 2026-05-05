# Practice: Text Preprocessing

## Concept Questions
1. Why is lowercasing generally the very first step in text preprocessing?
2. What is the difference between a tokenizer that splits by space (`text.split()`) and `nltk.word_tokenize()`?
3. Give an example of a situation where removing stopwords would be a **bad** idea.
4. Explain the difference between Stemming and Lemmatization to a non-technical person.

## Coding Tasks
1. **Basic Clean:** Write a Python function using base Python (no NLTK/SpaCy) that takes a string, makes it lowercase, removes commas and periods, and splits it by spaces.
2. **Stopword Filter:** Given the text `"The quick brown fox jumps over the lazy dog"`, use NLTK to tokenize it and remove all English stopwords. Print the resulting list.
3. **Stem vs Lemma Challenge:** Run the word `"organization"` through NLTK's `PorterStemmer` and `WordNetLemmatizer`. What are the outputs? Why are they different?

## Interpretation Tasks
Look at this output from a stemmer:
`['appl', 'comput', 'are', 'veri', 'expens']`
What do you think the original sentence was? What problems might this aggressive stemming cause if we try to read the data manually later?
