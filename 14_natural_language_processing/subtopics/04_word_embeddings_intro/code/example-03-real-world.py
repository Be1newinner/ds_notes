"""
Example 03: Text Classification with Embeddings
How to use pre-trained word vectors to train a standard scikit-learn classifier.
"""

import spacy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Loading SpaCy Model...")
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("ERROR: Please run `python -m spacy download en_core_web_md` in your terminal first!")
    exit()

# 1. Dataset
sentences = [
    "I absolutely loved the meal, it was delicious.",
    "The food was terrible and cold.",
    "Great service, highly recommend this place.",
    "I hated it, the waiter was rude.",
    "Awesome experience, five stars.",
    "Awful. Do not go here."
]
labels = [1, 0, 1, 0, 1, 0] # 1=Positive, 0=Negative

print("--- 1. Vectorizing with SpaCy ---")
# Instead of TfidfVectorizer, we ask SpaCy for the .vector of each sentence
# We create an empty list, then append the 300-dim numpy array for each sentence
X_vectors = []
for text in sentences:
    doc = nlp(text)
    X_vectors.append(doc.vector)

# Convert list of arrays into a 2D numpy matrix (X)
X = np.array(X_vectors)
y = np.array(labels)

print(f"Feature Matrix Shape: {X.shape} (Documents x 300 dimensions)")

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("\n--- 2. Training the Model ---")
# We just pass the dense arrays into a standard ML algorithm
model = LogisticRegression()
model.fit(X_train, y_train)

# 3. Evaluation
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100}%")

print("\n--- 3. Inference on New Data ---")
new_review = "The pasta was exquisite!"
# We must vectorize the new review the exact same way
new_vec = nlp(new_review).vector

# Sklearn expects a 2D array for prediction: .reshape(1, -1) handles this for a single sample
prediction = model.predict(new_vec.reshape(1, -1))
sentiment = "Positive" if prediction[0] == 1 else "Negative"

print(f"Review: '{new_review}'")
print(f"Prediction: {sentiment}")
print("Notice the model predicted Positive even though it never saw the words 'pasta' or 'exquisite' during training!")
