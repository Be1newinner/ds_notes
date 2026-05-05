# Examples: KNN and Naive Bayes

Here is a breakdown of the Python examples provided in the `code/` directory.

## 1. Basic KNN (`example-01-basic-knn.py`)
- **Goal:** Understand the mechanics of K-Nearest Neighbors and the critical importance of feature scaling.
- **Dataset:** A synthetic dataset where two features have vastly different scales (e.g., Income in \$100,000s vs Age in 10s).
- **Key Concepts Shown:** 
  - Training `KNeighborsClassifier`.
  - Comparing accuracy *before* and *after* applying `StandardScaler`.
- **Takeaway:** Unscaled KNN is completely dominated by features with large numeric ranges. Scaling is non-negotiable.

## 2. Text Classification with Naive Bayes (`example-02-naive-bayes-text.py`)
- **Goal:** Show the absolute best use-case for Naive Bayes: Natural Language Processing.
- **Dataset:** A tiny mock dataset of Spam vs Ham (Not Spam) text messages.
- **Key Concepts Shown:** 
  - Using `CountVectorizer` to turn text strings into a matrix of word counts.
  - Applying `MultinomialNB`.
  - Looking at how Laplace smoothing (`alpha`) allows the model to handle words it has never seen before.
- **Takeaway:** Naive Bayes is extremely fast and effective for baseline text classification models.

## 3. Real-World Scenario: Fraud Detection (`example-03-real-world-fraud.py`)
- **Goal:** Compare KNN and Gaussian Naive Bayes on a continuous dataset.
- **Dataset:** A simulated transaction dataset (Transaction Amount, Distance from Home) to predict Fraud.
- **Key Concepts Shown:** 
  - Using `GaussianNB` for continuous tabular data.
  - Tuning the `n_neighbors` parameter in KNN using a simple loop to find the "sweet spot" between overfitting and underfitting.
- **Takeaway:** Different algorithms shine in different scenarios; visualizing the accuracy across different $K$ values helps understand the Bias-Variance tradeoff.
