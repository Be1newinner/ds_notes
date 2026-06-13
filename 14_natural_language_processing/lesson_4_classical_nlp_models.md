# Classical NLP Models

## Learning Objective
Students should learn how to apply traditional machine learning algorithms (like Naive Bayes and Logistic Regression) to text data that has been preprocessed and vectorized.

## What Is This Topic?
Once text is converted into a numerical matrix (like a TF-IDF matrix), we can treat it just like any other dataset (like predicting house prices or customer churn). "Classical NLP" refers to using standard ML models rather than Deep Learning (Neural Networks).

## Why This Topic Matters
Deep learning models (like Transformers or GPT) require massive amounts of data, computing power, and time to train. For many everyday business problems—like filtering spam, routing support tickets, or basic sentiment analysis—Classical NLP models are much faster, cheaper, easier to explain, and often just as accurate.

## Core Intuition
If the word "win", "lottery", and "click" appear often in emails labeled as "Spam", a model will learn to associate the numerical columns representing those words with the output label "1" (Spam). 

## Key Concepts

### 1. The Modeling Pipeline
The absolute most critical concept in practical NLP is the Pipeline. You must ensure that the EXACT same preprocessing and vectorization steps applied to your training data are applied to your incoming real-world data.

### 2. Naive Bayes (MultinomialNB)
The undisputed king of classical text classification.
- **Why it works well:** It relies on probability. It calculates the probability of a document being Spam *given* the words it contains.
- **Why it's "Naive":** It assumes every word is independent of every other word. It assumes the presence of the word "free" has nothing to do with the word "money". Even though this is linguistically false, it works remarkably well in practice.

### 3. Logistic Regression & SVMs
These models also work exceptionally well on high-dimensional sparse text data. They often outperform Naive Bayes if you have a lot of training data.

### 4. Topic Modeling (Unsupervised NLP)
What if you don't have labels (like "Spam" or "Not Spam")? Topic modeling algorithms like Latent Dirichlet Allocation (LDA) can read a million documents and automatically group them into "topics" based on words that frequently occur together.

## Step-by-Step Explanation (Supervised Classification)
1. Split your raw text data into `X_train`, `X_test`, `y_train`, `y_test`.
2. Initialize a Vectorizer (e.g., `TfidfVectorizer`).
3. `fit_transform` the `X_train`.
4. Train a `MultinomialNB` model on the transformed `X_train` and `y_train`.
5. `transform` the `X_test`.
6. Predict using the trained model on the transformed `X_test`.
7. Evaluate using standard classification metrics (Accuracy, F1-Score).

## Real-World Uses
- Spam Filters.
- Sentiment Analysis on Product Reviews.
- Automatically tagging news articles (Sports, Politics, Tech).

## Advantages
- Extremely fast to train.
- Works well on small to medium-sized datasets.
- Highly interpretable (you can easily see which words the model thinks are most "spammy").

## Limitations
- Cannot understand context, sarcasm, or complex grammar (because Bag of Words destroys word order).

## Code References
- `code/example-01-naive-bayes-sentiment.py` — Building a basic sentiment classifier.
- `code/example-02-topic-modeling.py` — Finding hidden topics using LDA.
- `code/example-03-real-world-pipeline.py` — The professional way to build NLP models using `sklearn.pipeline.Pipeline`.


---

## Methods and Options: Classical NLP Models

### `sklearn.naive_bayes.MultinomialNB`
The standard model for text classification when features are counts or TF-IDF frequencies.

#### Syntax
`from sklearn.naive_bayes import MultinomialNB`
`model = MultinomialNB(alpha=1.0)`

#### Parameters
- `alpha`: Additive (Laplace/Lidstone) smoothing parameter. Default is 1.0. 
  - *Why it's needed:* If a word appears in the test set that was never seen in the training set for a specific class, the raw probability would be 0, ruining the entire calculation. `alpha` prevents 0 probabilities.

#### Common Methods
- `fit(X, y)`
- `predict(X)`
- `predict_proba(X)`: Returns the probability estimates for each class.
- `feature_log_prob_`: An attribute holding the empirical log probability of features given a class (used to interpret the model).

### `sklearn.pipeline.Pipeline`
**Crucial for NLP.** It chains multiple steps together so you don't leak data or forget a step during prediction.

#### Syntax
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])
```

#### Typical Workflow
1. `pipeline.fit(X_train_raw_text, y_train)`: Automatically fits the vectorizer, transforms the text, and trains the model.
2. `pipeline.predict(X_test_raw_text)`: Automatically transforms the new text using the learned vocabulary, then predicts.

### `sklearn.decomposition.LatentDirichletAllocation` (LDA)
Used for unsupervised Topic Modeling.

#### Parameters
- `n_components`: Number of topics you want the algorithm to find.
- `random_state`: For reproducibility.

---

## Examples for Classical NLP Models

This directory contains Python scripts demonstrating how to train machine learning models on text data.

### Code References
- `code/example-01-naive-bayes-sentiment.py` — A step-by-step approach to training a Naive Bayes classifier on vectorized text data.
- `code/example-02-topic-modeling.py` — An introduction to unsupervised NLP, finding topics in unlabelled text using LDA.
- `code/example-03-real-world-pipeline.py` — The industry-standard way to combine vectorization and modeling using `scikit-learn` Pipelines to prevent data leakage and simplify deployment.

---

## Practice: Classical NLP

### Concept Questions
1. Why is Naive Bayes considered "Naive"?
2. In NLP, why do we often prefer Multinomial Naive Bayes over Gaussian Naive Bayes?
3. Explain the purpose of a `Pipeline` in `scikit-learn` when working with text data. What mistake does it help prevent?
4. If you have a collection of 10,000 customer reviews but no labels indicating whether they are positive or negative, what technique could you use to understand what customers are talking about?

### Coding Tasks
1. **Pipeline Creation:** Write a Python snippet that creates a `Pipeline` combining a `CountVectorizer` (with English stopwords removed) and a `LogisticRegression` model.
2. **Train and Predict:** Using the pipeline from Task 1, fit it on `X_train = ["I loved it", "Terrible product"]`, `y_train = [1, 0]`. Then predict the sentiment of a new review: `["It was absolutely terrible"]`.

### Interpretation Tasks
You train a Topic Model (LDA) with `n_components=3`. You look at the top words for each topic:
- Topic 1: `[screen, battery, camera, charge, app]`
- Topic 2: `[flight, seat, delayed, luggage, attendant]`
- Topic 3: `[food, service, delicious, waiter, cold]`

How would you label these three topics in a business presentation?

---

## Interview Questions: Classical NLP

### Beginner Level
1. How does a machine learning model like Logistic Regression handle text data?
2. What is the difference between Supervised NLP (like Sentiment Analysis) and Unsupervised NLP (like Topic Modeling)?

### Intermediate Level
3. Walk me through the exact pipeline of steps required to build a spam classifier from raw text to prediction.
4. Why is Naive Bayes such a popular baseline model for text classification tasks?
5. What is data leakage in NLP, and how does using `fit_transform` on the entire dataset before doing `train_test_split` cause it?

### Advanced / Practical Level
6. You built a sentiment classifier using TF-IDF and Logistic Regression. Your accuracy on the test set is 95%. However, when deployed, it classifies the phrase "Not bad at all, I actually loved it" as NEGATIVE. Why did the model likely make this mistake, and how could you fix the vectorizer to potentially solve it? 
   *(Answer: It likely focused on "Not" and "bad" independently. Fixing it involves changing `ngram_range=(1,2)` to capture the bigram "Not bad").*
7. How would you determine the optimal number of topics (`n_components`) when using LDA for topic modeling?

---

## Python Code Examples

### `example-01-naive-bayes-sentiment.py`

```python
"""
Example 01: Naive Bayes Sentiment Classifier
A step-by-step approach without using Pipelines (to show the mechanics).
"""

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Dataset (Raw Text)
X = [
    "I absolutely loved this movie, it was fantastic!",
    "Terrible acting and a boring plot.",
    "Best film of the year, highly recommended.",
    "I hated every minute of it, awful.",
    "Great visuals and an amazing soundtrack.",
    "Total waste of money and time."
]
# 1 = Positive, 0 = Negative
y = [1, 0, 1, 0, 1, 0]

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 3. Vectorization
vectorizer = TfidfVectorizer(stop_words='english')

# FIT on train, TRANSFORM train
X_train_vec = vectorizer.fit_transform(X_train)

# ONLY TRANSFORM test (using the vocabulary learned from train)
X_test_vec = vectorizer.transform(X_test)

# 4. Modeling
model = MultinomialNB()
model.fit(X_train_vec, y_train) # Train the model

# 5. Prediction and Evaluation
y_pred = model.predict(X_test_vec)

print("--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# 6. Predicting new, unseen text
new_reviews = ["This was a great movie!", "Absolutely terrible, do not watch."]
# We MUST vectorize the new text before predicting!
new_reviews_vec = vectorizer.transform(new_reviews)
predictions = model.predict(new_reviews_vec)

print("\n--- New Predictions ---")
for text, pred in zip(new_reviews, predictions):
    sentiment = "Positive" if pred == 1 else "Negative"
    print(f"Review: '{text}' --> Prediction: {sentiment}")
```

### `example-02-topic-modeling.py`

```python
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
```

### `example-03-real-world-pipeline.py`

```python
"""
Example 03: The NLP Pipeline (Best Practice)
Using sklearn.pipeline to chain Vectorization and Modeling.
This prevents data leakage and makes deploying the model incredibly easy.
"""

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Dataset
X = [
    "The battery life on this phone is amazing.",
    "Customer service was terrible, they hung up on me.",
    "Highly recommend this restaurant, great food.",
    "The shoes fell apart after one week of use.",
    "Fast shipping and the product works as expected.",
    "Worst experience ever, do not buy."
]
y = [1, 0, 1, 0, 1, 0] # 1=Positive, 0=Negative

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("--- 1. Creating the Pipeline ---")
# The pipeline takes a list of (name, step_object) tuples.
nlp_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2))),
    ('classifier', LogisticRegression())
])

print("--- 2. Fitting the Pipeline ---")
# We pass the RAW TEXT directly to the pipeline.
# It automatically fits the vectorizer, transforms the text, and trains the Logistic Regression.
nlp_pipeline.fit(X_train, y_train)
print("Pipeline trained successfully.")

print("\n--- 3. Predicting with the Pipeline ---")
# We pass the RAW TEXT test data. 
# It automatically transforms using the learned vocab and predicts.
accuracy = nlp_pipeline.score(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100}%")

print("\n--- 4. Real World Inference ---")
new_data = [
    "This was a fantastic purchase!",
    "Broke immediately. Hate it."
]
# No need to manually call vectorizer.transform()! The pipeline handles it.
predictions = nlp_pipeline.predict(new_data)

for text, pred in zip(new_data, predictions):
    print(f"'{text}' -> {'Positive' if pred == 1 else 'Negative'}")
```
