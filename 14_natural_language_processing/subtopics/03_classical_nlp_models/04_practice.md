# Practice: Classical NLP

## Concept Questions
1. Why is Naive Bayes considered "Naive"?
2. In NLP, why do we often prefer Multinomial Naive Bayes over Gaussian Naive Bayes?
3. Explain the purpose of a `Pipeline` in `scikit-learn` when working with text data. What mistake does it help prevent?
4. If you have a collection of 10,000 customer reviews but no labels indicating whether they are positive or negative, what technique could you use to understand what customers are talking about?

## Coding Tasks
1. **Pipeline Creation:** Write a Python snippet that creates a `Pipeline` combining a `CountVectorizer` (with English stopwords removed) and a `LogisticRegression` model.
2. **Train and Predict:** Using the pipeline from Task 1, fit it on `X_train = ["I loved it", "Terrible product"]`, `y_train = [1, 0]`. Then predict the sentiment of a new review: `["It was absolutely terrible"]`.

## Interpretation Tasks
You train a Topic Model (LDA) with `n_components=3`. You look at the top words for each topic:
- Topic 1: `[screen, battery, camera, charge, app]`
- Topic 2: `[flight, seat, delayed, luggage, attendant]`
- Topic 3: `[food, service, delicious, waiter, cold]`

How would you label these three topics in a business presentation?
