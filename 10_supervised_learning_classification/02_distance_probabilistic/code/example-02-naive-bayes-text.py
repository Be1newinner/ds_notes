"""
Example 02: Text Classification with Multinomial Naive Bayes
Goal: Show why Naive Bayes is the standard baseline algorithm for NLP tasks like Spam detection.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. A tiny mock dataset of Text Messages
texts = [
    "Win a free iPhone now! Click here",          # Spam
    "Hey mom, when are you coming home?",         # Ham
    "Urgent! You have won $10,000 cash prize",    # Spam
    "Are we still on for lunch tomorrow?",        # Ham
    "Claim your free lottery ticket today",       # Spam
    "Please send me the project report ASAP",     # Ham
    "Limited offer! Buy one get one free",        # Spam
    "Don't forget to buy milk on the way back"    # Ham
]
labels = [1, 0, 1, 0, 1, 0, 1, 0] # 1 = Spam, 0 = Ham

# 2. Convert text to numbers using CountVectorizer
# Naive Bayes needs features to be word counts
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# X is now a matrix where each row is a message, and each column is a unique word count
print("Vocabulary learned by vectorizer:")
print(vectorizer.get_feature_names_out()[:10], "...\n")

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# 4. Train Multinomial Naive Bayes
# alpha=1.0 is Laplace smoothing. It prevents errors if the model sees a new word.
model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# 6. Predict on a completely new message
new_message = ["Congratulations, claim your free cash!"]
new_message_vectorized = vectorizer.transform(new_message)

prediction = model.predict(new_message_vectorized)
prob = model.predict_proba(new_message_vectorized)

class_name = "Spam" if prediction[0] == 1 else "Ham"
print(f"New Message: '{new_message[0]}'")
print(f"Prediction: {class_name} (Confidence: {prob[0][prediction[0]]:.4f})")
