# Code Examples: Content-Based Filtering

These examples demonstrate how to build a recommendation engine based on item characteristics and textual metadata.

## Code References

- **`code/example-01-basic.py`** — A simple example using short categorical text tags (like genres) and `CountVectorizer` to find similar items.
- **`code/example-02-intermediate.py`** — A more advanced example using full sentences (plot descriptions) and `TfidfVectorizer` to handle real language processing, combined with Cosine Similarity.
- **`code/example-03-real-world.py`** — Demonstrates how to build a User Profile vector by combining the vectors of multiple items a user has interacted with, allowing recommendations based on an entire user history rather than just a single item.
