# Method Options and Properties: Content-Based Filtering

To build a Content-Based Filtering system, we rely heavily on text vectorization and similarity metrics. Scikit-Learn provides the core tools.

## 1. Text Vectorization

### `sklearn.feature_extraction.text.TfidfVectorizer`
- **Purpose**: Converts a collection of raw documents (text) to a matrix of TF-IDF (Term Frequency-Inverse Document Frequency) features. It down-weights words that appear in many documents.
- **Syntax**: `vectorizer = TfidfVectorizer(stop_words='english')`
- **Common Arguments**:
  - `stop_words='english'`: Automatically removes common English words ("the", "is", "and"). Crucial for good similarity matching.
  - `max_features`: Limits the vocabulary size to the top N words (useful for memory saving).
  - `ngram_range`: Allows capturing phrases, e.g., `(1, 2)` captures single words and two-word combinations ("machine learning").
- **Typical Workflow**:
  ```python
  tf = TfidfVectorizer(stop_words='english')
  tfidf_matrix = tf.fit_transform(df['description'])
  ```

### `sklearn.feature_extraction.text.CountVectorizer`
- **Purpose**: Simpler alternative to TF-IDF. Just counts word occurrences. Useful for tag-based features (like genres separated by spaces or commas).
- **Return Type**: Both return a Scipy sparse matrix.

## 2. Similarity Calculation

### `sklearn.metrics.pairwise.cosine_similarity`
- **Purpose**: Computes the Cosine Similarity between all vectors in a matrix.
- **Syntax**: `cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)`
- **Core Intuition**: Calculates the cosine of the angle between two vectors. Independent of vector magnitude (length of document).
- **Return Type**: A dense Numpy array (matrix) where `matrix[i][j]` is the similarity between item `i` and item `j`.
- **Warning**: If you have 100,000 items, `cosine_similarity(matrix, matrix)` will create a 100,000 x 100,000 dense matrix in RAM, which requires ~40GB of memory. Use with caution on very large datasets!

### `sklearn.metrics.pairwise.linear_kernel`
- **Purpose**: A faster alternative to `cosine_similarity` ONLY if your vectors are already normalized (which TF-IDF vectors are by default).
- **Syntax**: `cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)`
- **Why use it?**: It mathematically results in the exact same output as cosine similarity when vectors are normalized, but runs slightly faster.

## 3. Retrieving Recommendations

### Pandas Series Indexing trick
- **Purpose**: To easily find the index of a movie by its title, so you can look up its row in the similarity matrix.
- **Code Pattern**:
  ```python
  # Create a reverse mapping from title to DataFrame index
  indices = pd.Series(df.index, index=df['title']).drop_duplicates()
  
  # Get index for 'The Matrix'
  idx = indices['The Matrix']
  ```

## Common Mistakes
- **Forgetting to drop missing text**: If a row has `NaN` instead of text, `TfidfVectorizer` will throw an error. Always use `df['text'].fillna('')` first.
- **Applying Cosine Similarity to the entire dataset at once in production**: In real-world systems, you rarely calculate the full NxN matrix if N is huge. You calculate similarity on the fly for a specific item, or use Approximate Nearest Neighbors (ANN) libraries like FAISS.
