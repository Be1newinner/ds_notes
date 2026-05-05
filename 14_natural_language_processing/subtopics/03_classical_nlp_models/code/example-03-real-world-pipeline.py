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
