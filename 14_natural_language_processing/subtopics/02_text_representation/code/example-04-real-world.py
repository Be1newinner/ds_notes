"""
Example 04: Real World Vectorization Workflow
Demonstrates Train/Test splits, min_df, max_df, and max_features.
"""

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample dataset
documents = [
    "Data science is the sexiest job.",
    "Python is great for data science.",
    "I love writing Python code.",
    "Machine learning models are powerful.",
    "Data data data everywhere.",
    "Deep learning is a subset of machine learning."
]
labels = [1, 1, 1, 0, 1, 0] # Fake labels

print("--- 1. Train Test Split ---")
# ALWAYS split before vectorizing!
X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.33, random_state=42)
print(f"Training docs: {len(X_train)}")
print(f"Testing docs: {len(X_test)}\n")

print("--- 2. Initialize Vectorizer with Constraints ---")
vectorizer = TfidfVectorizer(
    stop_words='english', # Remove stopwords
    max_features=10,      # Keep only top 10 words (prevents huge memory usage in real apps)
    min_df=2,             # Word must appear in at least 2 documents
    max_df=0.9            # Word must not appear in more than 90% of documents
)

print("--- 3. Fit and Transform on TRAIN ---")
# The vectorizer learns the vocabulary ONLY from the training data
X_train_vec = vectorizer.fit_transform(X_train)
print(f"Learned Vocabulary: {vectorizer.get_feature_names_out()}")
print(f"Train Matrix Shape: {X_train_vec.shape}")

print("\n--- 4. ONLY Transform on TEST ---")
# The vectorizer uses the already learned vocabulary to score the test data
# NEVER call .fit() or .fit_transform() on X_test!
X_test_vec = vectorizer.transform(X_test)
print(f"Test Matrix Shape: {X_test_vec.shape}")

print("\nCrucial Concept:")
print("If a word exists in X_test but was not seen in X_train (or didn't meet min_df), it is completely ignored. This simulates real life where a deployed model encounters words it has never seen before.")
