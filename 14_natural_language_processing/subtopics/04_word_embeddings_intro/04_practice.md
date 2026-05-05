# Practice: Word Embeddings

## Concept Questions
1. What is the difference between a sparse vector (TF-IDF) and a dense vector (Word2Vec)?
2. If two words appear in similar contexts (e.g., "I drink coffee" and "I drink tea"), what will happen to their Word2Vec representations during training?
3. Why do we almost always download pre-trained embeddings (like GloVe or SpaCy's models) instead of training our own from scratch?

## Coding Tasks
1. **Gensim Training:** Create a tiny corpus: `[["the", "king", "rules"], ["the", "queen", "rules"]]`. Train a `Word2Vec` model using `gensim` on this data with `vector_size=5`. Print the vector for "king".
2. **SpaCy Similarity:** Load `en_core_web_md` using `spacy`. Calculate and print the similarity between the sentence "I want to buy a car" and "I am looking to purchase an automobile". (Even though they share few words, the similarity should be high).

## Interpretation Tasks
You are building a search engine for a medical database. A doctor searches for "heart attack". The database has a great article titled "Myocardial Infarction Symptoms". 
If you used TF-IDF, would the article show up in the search results? Why or why not? 
If you used Word Embeddings to compare the search query to the article titles, would it show up? Why?
