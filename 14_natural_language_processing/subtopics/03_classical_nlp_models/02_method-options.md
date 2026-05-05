# Methods and Options: Classical NLP Models

## `sklearn.naive_bayes.MultinomialNB`
The standard model for text classification when features are counts or TF-IDF frequencies.

### Syntax
`from sklearn.naive_bayes import MultinomialNB`
`model = MultinomialNB(alpha=1.0)`

### Parameters
- `alpha`: Additive (Laplace/Lidstone) smoothing parameter. Default is 1.0. 
  - *Why it's needed:* If a word appears in the test set that was never seen in the training set for a specific class, the raw probability would be 0, ruining the entire calculation. `alpha` prevents 0 probabilities.

### Common Methods
- `fit(X, y)`
- `predict(X)`
- `predict_proba(X)`: Returns the probability estimates for each class.
- `feature_log_prob_`: An attribute holding the empirical log probability of features given a class (used to interpret the model).

## `sklearn.pipeline.Pipeline`
**Crucial for NLP.** It chains multiple steps together so you don't leak data or forget a step during prediction.

### Syntax
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])
```

### Typical Workflow
1. `pipeline.fit(X_train_raw_text, y_train)`: Automatically fits the vectorizer, transforms the text, and trains the model.
2. `pipeline.predict(X_test_raw_text)`: Automatically transforms the new text using the learned vocabulary, then predicts.

## `sklearn.decomposition.LatentDirichletAllocation` (LDA)
Used for unsupervised Topic Modeling.

### Parameters
- `n_components`: Number of topics you want the algorithm to find.
- `random_state`: For reproducibility.
