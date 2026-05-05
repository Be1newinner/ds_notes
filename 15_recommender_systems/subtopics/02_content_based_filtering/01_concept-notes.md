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
