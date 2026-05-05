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
