# Text Representation Methods and Options

We primarily use `scikit-learn` for classical text representation.

## `sklearn.feature_extraction.text.CountVectorizer`
Converts a collection of text documents to a matrix of token counts (Bag of Words).

### Syntax
`vectorizer = CountVectorizer(stop_words='english', max_features=1000)`

### Common Parameters
- `stop_words='english'`: Automatically removes English stopwords (though doing it manually in preprocessing is often safer).
- `lowercase=True`: Defaults to True. Automatically lowercases text.
- `max_features`: Int. Limits the vocabulary to the top `N` most frequent words. Crucial for memory management!
- `ngram_range`: Tuple `(min_n, max_n)`. e.g., `(1, 2)` means unigrams and bigrams.
- `min_df`: Float or Int. Minimum document frequency. Ignore terms that have a document frequency strictly lower than this threshold (removes extremely rare words / typos).
- `max_df`: Float or Int. Maximum document frequency. Ignore terms that appear in more than this % of documents (removes corpus-specific stopwords).

### Common Methods
- `fit(X)`: Learns the vocabulary dictionary of all tokens in the raw documents.
- `transform(X)`: Transforms documents to document-term matrix.
- `fit_transform(X)`: Does both in one step. (Use on training data!).
- `get_feature_names_out()`: Returns the list of words that correspond to the columns of the matrix.

## `sklearn.feature_extraction.text.TfidfVectorizer`
Convert a collection of raw documents to a matrix of TF-IDF features.

### Syntax
`vectorizer = TfidfVectorizer(max_features=1000)`

### Common Parameters
It inherits almost all parameters from `CountVectorizer` (`max_features`, `ngram_range`, `min_df`, `max_df`).
- `norm`: 'l1', 'l2', or None. Defaults to 'l2' (Normalizes the vectors to have length 1, making cosine similarity easier).

### Typical Workflow
1. Split data into Train and Test sets.
2. Initialize Vectorizer.
3. `X_train_vec = vectorizer.fit_transform(X_train)` (Learn vocab AND transform).
4. `X_test_vec = vectorizer.transform(X_test)` (ONLY transform based on train vocab. NEVER fit on test data!).

## Understanding the Output (Sparse Matrices)
The output of `fit_transform` is a **SciPy Sparse Matrix**, not a Pandas DataFrame or NumPy array. 
Because 99% of the matrix is zeros, SciPy only stores the locations of the non-zero values to save RAM.
- Use `.toarray()` to convert it to a NumPy array (only do this for small datasets or viewing purposes, otherwise your RAM will crash!).
