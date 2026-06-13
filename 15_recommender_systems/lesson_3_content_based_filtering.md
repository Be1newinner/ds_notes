# Content-Based Filtering

## Learning Objective
What students should understand after this lesson:
- The fundamental idea behind content-based filtering.
- How to represent items as feature vectors (text, genres, tags).
- How to measure similarity between items (Cosine Similarity).
- How to build a user profile based on the items they like.

## What Is This Topic?
Content-Based Filtering recommends items by comparing the *content* or *features* of the items to a profile of the user's preferences. If you watch a lot of Sci-Fi movies directed by Christopher Nolan, the system will recommend other Sci-Fi movies by Christopher Nolan.

## Why This Topic Matters
Content-based methods are excellent for overcoming the "Item Cold Start" problem. Even if a movie was released 5 minutes ago and nobody has watched it, a content-based system can still recommend it to users who like its genre, actors, and director.

## Core Intuition
Imagine you tell a librarian: "I loved *The Lord of the Rings*. It has magic, elves, and an epic quest." The librarian thinks, "What other books have magic, elves, and quests? Ah, *The Hobbit* or *Eragon*!" The system does the same thing mathematically by finding items with matching "tags".

## Key Concepts
- **Item Profile**: A set of features describing the item (e.g., genre, cast, description text).
- **User Profile**: A vector describing the user's preferences, usually created by averaging the item profiles of things they have liked.
- **Vectorization**: Converting text descriptions into numbers (e.g., TF-IDF).
- **Cosine Similarity**: A math trick to measure how close two vectors are, irrespective of their size. It measures the angle between them. 1 means identical, 0 means completely different.

## Step-by-Step Explanation
1. **Extract Features**: Gather metadata about all items (e.g., genre tags, plot summaries).
2. **Vectorize Items**: Convert the metadata into a mathematical vector (an array of numbers). For text, use TF-IDF or Count Vectorizer.
3. **Calculate Similarity**: Create an Item-Item similarity matrix where every item is compared to every other item using Cosine Similarity.
4. **Recommend**: When a user looks at Item X, look up Item X in the similarity matrix, and return the items with the highest similarity scores.

## Important Parameters / Options / Settings
- **Features to Include**: Choosing the right features is critical. Recommending a movie purely on "Year Released" will give bad results. Recommending based on "Plot keywords" is much better.
- **Vectorization Method**: TF-IDF (gives less weight to common words like "the") vs. Count Vectorizer (pure counts).
- **Similarity Metric**: Cosine Similarity is the industry standard because it handles documents of different lengths well.

## Output / Result Interpretation
The output is a list of items sorted by their similarity score to a target item. A score of 0.95 means highly similar, while 0.10 means practically unrelated.

## Real-World Uses
- **News Articles**: Recommending similar news stories based on the text of the article the user is currently reading.
- **Pandora Radio**: Recommending songs based on the "Music Genome Project" (musical traits of the songs).

## Advantages
- **No Item Cold Start**: Can recommend brand new items immediately.
- **Transparency**: Easy to explain ("We recommended this because you watched X").
- **Niche Interests**: Good at recommending items to users with unique tastes, unlike Collaborative Filtering which pushes popular items.

## Limitations
- **Feature Engineering Heavy**: You must have good, clean metadata for every item.
- **No Serendipity**: It creates a "filter bubble." If you only watch Action movies, it will *never* recommend a Comedy, even an amazing one. It lacks the ability to discover hidden, unexpected interests.

## Common Mistakes
- **Using Euclidean Distance instead of Cosine Similarity for text**: Long plot descriptions will be artificially placed far away from short descriptions.
- **Ignoring Stop Words**: If you don't remove words like "and", "the", "is", your similarities will be based on grammar, not content.

## Related Methods
- TF-IDF (Natural Language Processing module)
- Collaborative Filtering (recommends based on user behavior rather than item content).

## Code References
- `code/example-01-basic.py` — Recommending based on simple categorical tags (Genres).
- `code/example-02-intermediate.py` — Recommending based on Text using TF-IDF and Cosine Similarity.
- `code/example-03-real-world.py` — Building User Profiles to recommend multiple items at once.


---

## Method Options and Properties: Content-Based Filtering

To build a Content-Based Filtering system, we rely heavily on text vectorization and similarity metrics. Scikit-Learn provides the core tools.

### 1. Text Vectorization

#### `sklearn.feature_extraction.text.TfidfVectorizer`
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

#### `sklearn.feature_extraction.text.CountVectorizer`
- **Purpose**: Simpler alternative to TF-IDF. Just counts word occurrences. Useful for tag-based features (like genres separated by spaces or commas).
- **Return Type**: Both return a Scipy sparse matrix.

### 2. Similarity Calculation

#### `sklearn.metrics.pairwise.cosine_similarity`
- **Purpose**: Computes the Cosine Similarity between all vectors in a matrix.
- **Syntax**: `cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)`
- **Core Intuition**: Calculates the cosine of the angle between two vectors. Independent of vector magnitude (length of document).
- **Return Type**: A dense Numpy array (matrix) where `matrix[i][j]` is the similarity between item `i` and item `j`.
- **Warning**: If you have 100,000 items, `cosine_similarity(matrix, matrix)` will create a 100,000 x 100,000 dense matrix in RAM, which requires ~40GB of memory. Use with caution on very large datasets!

#### `sklearn.metrics.pairwise.linear_kernel`
- **Purpose**: A faster alternative to `cosine_similarity` ONLY if your vectors are already normalized (which TF-IDF vectors are by default).
- **Syntax**: `cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)`
- **Why use it?**: It mathematically results in the exact same output as cosine similarity when vectors are normalized, but runs slightly faster.

### 3. Retrieving Recommendations

#### Pandas Series Indexing trick
- **Purpose**: To easily find the index of a movie by its title, so you can look up its row in the similarity matrix.
- **Code Pattern**:
  ```python
  # Create a reverse mapping from title to DataFrame index
  indices = pd.Series(df.index, index=df['title']).drop_duplicates()
  
  # Get index for 'The Matrix'
  idx = indices['The Matrix']
  ```

### Common Mistakes
- **Forgetting to drop missing text**: If a row has `NaN` instead of text, `TfidfVectorizer` will throw an error. Always use `df['text'].fillna('')` first.
- **Applying Cosine Similarity to the entire dataset at once in production**: In real-world systems, you rarely calculate the full NxN matrix if N is huge. You calculate similarity on the fly for a specific item, or use Approximate Nearest Neighbors (ANN) libraries like FAISS.

---

## Code Examples: Content-Based Filtering

These examples demonstrate how to build a recommendation engine based on item characteristics and textual metadata.

### Code References

- **`code/example-01-basic.py`** — A simple example using short categorical text tags (like genres) and `CountVectorizer` to find similar items.
- **`code/example-02-intermediate.py`** — A more advanced example using full sentences (plot descriptions) and `TfidfVectorizer` to handle real language processing, combined with Cosine Similarity.
- **`code/example-03-real-world.py`** — Demonstrates how to build a User Profile vector by combining the vectors of multiple items a user has interacted with, allowing recommendations based on an entire user history rather than just a single item.

---

## Practice: Content-Based Filtering

### Exercise 1: Understanding Vectors
You have three movies with the following genre tags:
- Movie A: "Action Sci-Fi"
- Movie B: "Action Adventure Sci-Fi"
- Movie C: "Comedy Romance"

Assume we use a simple Count Vectorizer with the vocabulary: `['Action', 'Adventure', 'Comedy', 'Romance', 'Sci-Fi']`

1. Write out the numeric vector for Movie A, Movie B, and Movie C.
2. Based on intuition (or math if you want), which two movies will have the highest Cosine Similarity?

### Exercise 2: TF-IDF Intuition
You are building a content-based recommender for news articles. Article 1 is about "Stock Market Trends". Article 2 is about "Tech Company Earnings".
Both articles contain the words "the", "and", "money" very frequently.
Why would `TfidfVectorizer` be better to use here than `CountVectorizer`?

### Exercise 3: Python Coding Challenge
Load a small dataset of books with a `description` column.
1. Use `TfidfVectorizer` (with English stop words removed) to convert the descriptions into a matrix.
2. Calculate the cosine similarity matrix.
3. Write a function `recommend_book(title)` that prints the top 3 most similar books based on the matrix.

### Exercise 4: The Filter Bubble
A user only watches documentaries about World War II. They rate all of them 5 stars.
1. Explain what a content-based recommender will suggest to them next.
2. What is the main drawback of this approach in this specific scenario?

---

## Interview Questions: Content-Based Filtering

### Beginner Questions
1. **Explain Content-Based Filtering in one sentence.**
   *Answer*: It recommends items to a user that are similar in features or content to items the user has liked in the past.
2. **What is Cosine Similarity?**
   *Answer*: It's a metric that measures the cosine of the angle between two vectors. It determines how similar two items are, ranging from 1 (identical) to 0 (completely different).

### Conceptual Questions
3. **Why do we use TF-IDF instead of simple word counts for text-based recommendations?**
   *Answer*: Because simple word counts give too much importance to frequent words like "the" or "is". TF-IDF penalizes words that appear across all documents, highlighting unique words that actually describe the specific content of an item.
4. **How does Content-Based Filtering solve the Item Cold Start problem?**
   *Answer*: Since it relies purely on item metadata (like genres, descriptions, authors), it does not need any user interaction history to understand the item. A brand new item can instantly be matched to users with similar tastes.

### Practical Questions
5. **What is a major disadvantage of Content-Based Filtering?**
   *Answer*: It suffers from "overspecialization" or the "filter bubble" effect. It can only recommend items similar to what the user has already seen. It cannot help a user discover entirely new genres or tastes outside their established profile.
6. **You have 1 million articles. Can you run `cosine_similarity()` on all of them to find recommendations?**
   *Answer*: No, computing the pairwise similarity of 1 million items results in a 1 trillion element matrix, which will cause an Out-Of-Memory (OOM) error. Instead, you would calculate similarity on the fly for a specific query, or use Approximate Nearest Neighbor (ANN) search algorithms like FAISS or Annoy.

---

## Python Code Examples

### `example-01-basic.py`

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example 01: Simple Tag-Based Recommender
# Recommending movies based purely on their Genre tags.

print("--- Simple Tag-Based Content Recommender ---")

# 1. Mock dataset of movies and their genres
data = {
    'movie_title': ['The Matrix', 'Inception', 'Toy Story', 'Finding Nemo', 'Terminator'],
    'genres': ['Action Sci-Fi', 'Action Sci-Fi Thriller', 'Animation Comedy Family', 'Animation Family', 'Action Sci-Fi']
}
df = pd.DataFrame(data)

# 2. Vectorize the text tags using CountVectorizer
# We use CountVectorizer because these are simple tags, not complex natural language sentences.
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['genres'])

print("Vocabulary Learned:")
print(cv.get_feature_names_out())
print("\nVectorized Matrix (Dense representation):")
print(count_matrix.toarray())
print("\n")

# 3. Calculate Cosine Similarity between all movies
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# 4. Create a recommendation function
def recommend(title, df, sim_matrix):
    # Get the index of the movie that matches the title
    idx = df[df['movie_title'] == title].index[0]
    
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(sim_matrix[idx]))
    
    # Sort the movies based on the similarity scores (descending)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 2 most similar movies (excluding itself)
    sim_scores = sim_scores[1:3]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    
    # Print results
    print(f"Recommendations for '{title}':")
    for i, idx in enumerate(movie_indices):
        print(f"- {df['movie_title'].iloc[idx]} (Similarity Score: {scores[i]:.2f})")

# Let's test it!
recommend('The Matrix', df, cosine_sim)
```

### `example-02-intermediate.py`

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Example 02: Text-Based Recommender (TF-IDF)
# Using Natural Language plot descriptions to recommend items.

print("--- Text-Based Content Recommender (TF-IDF) ---")

# 1. Mock dataset of movies and their Plot Descriptions
data = {
    'movie_title': ['The Dark Knight', 'Batman Begins', 'The Avengers', 'Superman', 'A Beautiful Mind'],
    'plot': [
        'Batman fights the Joker in Gotham City to save the people.',
        'Bruce Wayne trains to become Batman and saves Gotham City from destruction.',
        'Superheroes team up to save the world from an alien invasion.',
        'An alien orphan becomes a superhero to save Earth.',
        'A brilliant mathematician struggles with schizophrenia.'
    ]
}
df = pd.DataFrame(data)

# 2. Vectorize the text using TF-IDF
# We use stop_words='english' to remove 'the', 'a', 'to', etc.
tfidf = TfidfVectorizer(stop_words='english')

# fit_transform learns the vocabulary and transforms the text into vectors
tfidf_matrix = tfidf.fit_transform(df['plot'])

print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
# Shows (5 movies, X unique words)

# 3. Calculate Cosine Similarity
# Because TF-IDF vectors are already normalized by sklearn, we can use linear_kernel
# which is mathematically equivalent to cosine_similarity but faster.
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# 4. Recommendation Function (Using Pandas indexing)
# Create a reverse mapping of indices and movie titles
indices = pd.Series(df.index, index=df['movie_title']).drop_duplicates()

def recommend_text(title, df, sim_matrix):
    idx = indices[title]
    
    # Get pairwise similarity scores
    sim_scores = list(enumerate(sim_matrix[idx]))
    
    # Sort them
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top 2
    sim_scores = sim_scores[1:3]
    
    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    
    print(f"\nRecommendations for '{title}':")
    for i, m_idx in enumerate(movie_indices):
         print(f"- {df['movie_title'].iloc[m_idx]} (Sim: {scores[i]:.2f})")

# Let's see what happens if we like Batman
recommend_text('The Dark Knight', df, cosine_sim)

# Let's see what happens if we like The Avengers
recommend_text('The Avengers', df, cosine_sim)
```

### `example-03-real-world.py`

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example 03: Building a User Profile Vector
# Instead of recommending based on a single movie, we build an "average" profile
# of what the user likes based on their history.

print("--- User Profile Content Recommender ---")

# 1. Dataset of articles
articles = pd.DataFrame({
    'article_id': [0, 1, 2, 3, 4],
    'title': ['AI in Healthcare', 'Stock Market drops', 'Machine Learning trends', 'Sports results', 'Deep Learning for Medical Imaging'],
    'text': [
        'AI is revolutionizing healthcare and hospitals.',
        'The stock market saw a massive drop today.',
        'Machine learning trends show growth in AI tech.',
        'Local sports team wins the championship.',
        'Deep learning models are reading medical x-rays.'
    ]
})

# 2. Vectorize Articles
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(articles['text'])

# 3. Simulate a User's History
# Let's say User A read and liked article 0 and article 2.
user_history_indices = [0, 2] 

print(f"User liked: '{articles['title'].iloc[0]}' and '{articles['title'].iloc[2]}'")

# 4. Build the User Profile Vector
# We grab the mathematical vectors for the articles they liked
liked_vectors = tfidf_matrix[user_history_indices]

# We average them together to create a single vector representing their taste
# (We use mean across axis=0 to average the columns)
user_profile_vector = np.asarray(liked_vectors.mean(axis=0))

# 5. Find recommendations
# We calculate the cosine similarity between the NEW user profile vector and ALL articles
sim_scores = cosine_similarity(user_profile_vector, tfidf_matrix)

# 6. Rank the articles
# sim_scores is a 2D array, we flatten it to 1D
sim_scores_1d = sim_scores.flatten()

# Get indices sorted by score, descending
sorted_indices = sim_scores_1d.argsort()[::-1]

print("\nTop Recommendations for User Profile:")
for idx in sorted_indices:
    # Don't recommend articles they already read
    if idx not in user_history_indices:
        print(f"- {articles['title'].iloc[idx]} (Score: {sim_scores_1d[idx]:.2f})")
```
