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
