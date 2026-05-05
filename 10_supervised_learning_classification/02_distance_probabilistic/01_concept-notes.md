# Distance and Probabilistic Methods: KNN and Naive Bayes

## Learning Objective
Understand the intuition, math, and practical applications of K-Nearest Neighbors (a distance-based method) and Naive Bayes (a probabilistic method) for classification.

## Part 1: K-Nearest Neighbors (KNN)

### What Is This Topic?
KNN is an algorithm that classifies a new data point based on how its neighbors are classified. "Tell me who your friends are, and I will tell you who you are."

### Core Intuition
Imagine a scatter plot of red dots and blue dots. If you drop a new, unknown dot into the plot, you look at its $K$ closest neighbors. If the majority of those $K$ neighbors are red, you classify the new dot as red.

### Key Concepts
- **Distance Metric**: How we measure "closeness" (usually Euclidean distance: $\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$).
- **$K$ (Number of Neighbors)**: The number of nearest points to look at. If $K=3$, we look at the 3 closest points.
- **Lazy Learning**: KNN doesn't actually "train" a mathematical model. It just memorizes the training dataset and does all the computational work during the prediction phase.

### Important Settings
- **`n_neighbors` ($K$)**: A small $K$ (e.g., 1) captures noise and overfits. A large $K$ (e.g., 100) smoothes the decision boundary and might underfit. $K$ is typically chosen as an odd number to prevent ties.

### Advantages & Limitations of KNN
- **Advantages**: Incredibly simple to understand; naturally handles multi-class problems; no training phase.
- **Limitations**: Very slow to predict on large datasets; terrible performance on high-dimensional data (the "curse of dimensionality"); **highly sensitive to unscaled data**.

---

## Part 2: Naive Bayes

### What Is This Topic?
Naive Bayes is a classification algorithm based on Bayes' Theorem. It calculates the probability of each class given the input features and selects the class with the highest probability.

### Core Intuition
Bayes' Theorem updates our beliefs based on new evidence. If we are trying to predict if an email is Spam, we start with the base probability of an email being Spam. Then, if we see the word "Viagra", we update that probability upward. 

### Key Concepts
- **Bayes' Theorem**: $P(A|B) = \frac{P(B|A) * P(A)}{P(B)}$
- **The "Naive" Assumption**: The algorithm assumes that all features (e.g., all words in an email) are completely independent of each other. In reality, words like "San" and "Francisco" are highly dependent, but the algorithm ignores this. Surprisingly, it still works very well.

### Important Variations
- **Multinomial Naive Bayes**: Good for discrete data (like word counts in text classification).
- **Gaussian Naive Bayes**: Good for continuous data; assumes features follow a normal (Gaussian) distribution.

### Advantages & Limitations of Naive Bayes
- **Advantages**: Extremely fast to train and predict; works exceptionally well on text data (NLP) and high-dimensional sparse data; requires relatively little training data.
- **Limitations**: The "naive" assumption is almost always false in real life; predicted probabilities are often poorly calibrated (they are often extremely close to 0 or 1).

## Real-World Uses
- **KNN**: Recommender systems (finding users similar to you), simple image recognition.
- **Naive Bayes**: Spam filtering, sentiment analysis, document categorization.

## Code References
- `code/example-01-basic-knn.py`
- `code/example-02-naive-bayes-text.py`
- `code/example-03-real-world-fraud.py`
