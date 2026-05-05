"""
Example 02: Topic Modeling (Unsupervised NLP)
Finding hidden topics in a collection of text using Latent Dirichlet Allocation (LDA).
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 1. Unlabelled Dataset
documents = [
    # Tech/Computers
    "Apple releases new iPhone with better battery.",
    "Microsoft Windows update causes computer crashes.",
    "The new laptop screen is high resolution.",
    # Sports
    "The team won the championship game last night.",
    "Player injured his knee during the football match.",
    "The referee made a terrible call in the soccer game."
]

print("--- 1. Vectorizing the Text ---")
# For LDA, CountVectorizer (raw counts) usually works better than TF-IDF
vectorizer = CountVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

print("--- 2. Applying LDA (Topic Modeling) ---")
# We tell the model we want to find 2 topics
lda_model = LatentDirichletAllocation(n_components=2, random_state=42)
lda_model.fit(X_vec)

print("\n--- 3. Interpreting the Topics ---")
# A topic is represented as a distribution over words. 
# We print the top words for each topic to understand what the topic is about.

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        # Sort words by importance for this topic and get the top N
        top_word_indices = topic.argsort()[:-no_top_words - 1:-1]
        print(" ".join([feature_names[i] for i in top_word_indices]))

# Display top 4 words per topic
display_topics(lda_model, feature_names, 4)

print("\nAnalysis:")
print("Notice how the algorithm successfully separated the 'Tech' words from the 'Sports' words without any labels!")
