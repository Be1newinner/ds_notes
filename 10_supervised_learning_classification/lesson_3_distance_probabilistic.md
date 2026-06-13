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


---

## Method Options: KNN and Naive Bayes in Scikit-Learn

### 1. K-Nearest Neighbors Classifier

#### Syntax
```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
```

#### Common Arguments
- **`n_neighbors`** (`int`, default=`5`): The number of neighbors to use by default for queries. This is the crucial hyperparameter to tune.
- **`weights`** (`{'uniform', 'distance'}`, default=`'uniform'`): 
  - `'uniform'`: All points in each neighborhood are weighted equally.
  - `'distance'`: Weight points by the inverse of their distance. Closer neighbors of a query point will have a greater influence than neighbors which are further away.
- **`metric`** (`str`, default=`'minkowski'`): The distance metric to use.
- **`p`** (`int`, default=`2`): Power parameter for the Minkowski metric. When `p=1`, this is equivalent to Manhattan distance. When `p=2`, this is Euclidean distance.

#### Typical Workflow & Mistakes
- **Mandatory Step**: You **must** scale your features (e.g., using `StandardScaler`) before using KNN. Because it relies on physical distance calculations, a feature ranging from 0-1000 will overpower a feature ranging from 0-1.
- **Mistake**: Using KNN on a dataset with millions of rows. It has to calculate the distance to every single row for every prediction. It will be too slow.

---

### 2. Naive Bayes Classifiers

Scikit-learn provides different Naive Bayes classifiers depending on the distribution of your data.

#### Gaussian Naive Bayes (For Continuous Data)
```python
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
```
- **Use Case**: When your features are continuous (e.g., height, weight, salary) and you can assume they are roughly normally distributed.
- **Arguments**: Very few arguments to tune. It's essentially a plug-and-play baseline model.

#### Multinomial Naive Bayes (For Discrete/Count Data)
```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB(alpha=1.0)
```
- **Use Case**: Text classification where features represent word counts (e.g., from `CountVectorizer`).
- **Arguments**:
  - **`alpha`** (`float`, default=`1.0`): Additive (Laplace/Lidstone) smoothing parameter. Set to 0 for no smoothing. This prevents the model from predicting a probability of exactly 0 if it encounters a word in the test set that wasn't in the training set.

#### Typical Workflow & Mistakes
- **Workflow for Text**: Use `CountVectorizer` or `TfidfVectorizer` to convert text to numbers, then fit `MultinomialNB`.
- **Mistake**: Using GaussianNB on sparse text data, or using MultinomialNB on continuous data containing negative values. MultinomialNB requires non-negative feature values.

---

## Examples: KNN and Naive Bayes

Here is a breakdown of the Python examples provided in the `code/` directory.

### 1. Basic KNN (`example-01-basic-knn.py`)
- **Goal:** Understand the mechanics of K-Nearest Neighbors and the critical importance of feature scaling.
- **Dataset:** A synthetic dataset where two features have vastly different scales (e.g., Income in \$100,000s vs Age in 10s).
- **Key Concepts Shown:** 
  - Training `KNeighborsClassifier`.
  - Comparing accuracy *before* and *after* applying `StandardScaler`.
- **Takeaway:** Unscaled KNN is completely dominated by features with large numeric ranges. Scaling is non-negotiable.

### 2. Text Classification with Naive Bayes (`example-02-naive-bayes-text.py`)
- **Goal:** Show the absolute best use-case for Naive Bayes: Natural Language Processing.
- **Dataset:** A tiny mock dataset of Spam vs Ham (Not Spam) text messages.
- **Key Concepts Shown:** 
  - Using `CountVectorizer` to turn text strings into a matrix of word counts.
  - Applying `MultinomialNB`.
  - Looking at how Laplace smoothing (`alpha`) allows the model to handle words it has never seen before.
- **Takeaway:** Naive Bayes is extremely fast and effective for baseline text classification models.

### 3. Real-World Scenario: Fraud Detection (`example-03-real-world-fraud.py`)
- **Goal:** Compare KNN and Gaussian Naive Bayes on a continuous dataset.
- **Dataset:** A simulated transaction dataset (Transaction Amount, Distance from Home) to predict Fraud.
- **Key Concepts Shown:** 
  - Using `GaussianNB` for continuous tabular data.
  - Tuning the `n_neighbors` parameter in KNN using a simple loop to find the "sweet spot" between overfitting and underfitting.
- **Takeaway:** Different algorithms shine in different scenarios; visualizing the accuracy across different $K$ values helps understand the Bias-Variance tradeoff.

---

## Practice Exercises: KNN and Naive Bayes

These exercises are designed to test your conceptual understanding and coding skills.

### Conceptual Questions
1. If you set $K = 1$ in K-Nearest Neighbors, what happens to the training accuracy? What will likely happen to the testing accuracy? Explain why in terms of overfitting.
2. Why is Naive Bayes called "Naive"? Provide a real-world example of where its core assumption fails.
3. You have a dataset with 50 million rows. Why might KNN be a terrible choice for this problem in a real-time production environment?

### Coding Tasks

#### Task 1: Finding the Best K
1. Load the `wine` dataset from `sklearn.datasets` (`load_wine()`).
2. Split the data into train and test sets (80/20).
3. Scale the features using `StandardScaler`.
4. Write a `for` loop that trains a `KNeighborsClassifier` for every odd value of $K$ from 1 to 21.
5. Store the test accuracy for each $K$ and print out which $K$ gave the best result.

#### Task 2: Gaussian Naive Bayes Baseline
Using the exact same scaled `wine` dataset from Task 1:
1. Initialize a `GaussianNB` model.
2. Train it and predict on the test set.
3. Compare the accuracy of Naive Bayes to the best accuracy you got from KNN. Which one performed better?

#### Task 3: Text Classification Pipeline
1. Create a tiny dataset of 5 sentences about sports (Label: 1) and 5 sentences about politics (Label: 0).
2. Use `sklearn.feature_extraction.text.CountVectorizer` to transform the text.
3. Train a `MultinomialNB` classifier.
4. Write a brand new sentence that mixes sports and politics (e.g., "The president threw a football"). Predict its class and print the `predict_proba` to see how confused the model is.

---

## Interview Questions: KNN and Naive Bayes

### Beginner Questions
1. **Explain K-Nearest Neighbors in one sentence.**
   *Hint:* It classifies a new data point by looking at the majority class among its $K$ closest neighbors in the training data.
2. **Does KNN require a training step?**
   *Hint:* No, it is a "lazy learner." The "training" phase is essentially just memorizing the dataset. All computation happens at prediction time.
3. **What is the difference between Gaussian and Multinomial Naive Bayes?**
   *Hint:* Gaussian is for continuous numeric data (assumes a normal distribution). Multinomial is for discrete counts (like word frequencies in text).

### Conceptual Questions
4. **How do you choose the right value for $K$ in KNN? What happens if $K$ is too small or too large?**
   *Hint:* You choose $K$ using cross-validation. If $K$ is too small (e.g., $K=1$), the model captures noise and overfits. If $K$ is too large (e.g., $K=N$), the model underfits and simply predicts the majority class of the entire dataset.
5. **Why is feature scaling absolutely critical for KNN?**
   *Hint:* KNN relies on calculating physical distances (like Euclidean distance). If one feature is measured in millions (e.g., salary) and another in single digits (e.g., years of experience), the distance metric will completely ignore the smaller feature.
6. **Explain the "Naive" assumption in Naive Bayes.**
   *Hint:* It assumes that all features are conditionally independent of each other given the class label. This is rarely true in reality, but the algorithm still performs well.

### Practical Questions
7. **In a Natural Language Processing task (like spam classification), a word appears in the test set that was never seen in the training set. How does Naive Bayes handle this?**
   *Hint:* Without intervention, the probability of that word would be 0, causing the entire posterior probability to multiply to 0. We use Laplace Smoothing (the `alpha` parameter) to assign a tiny non-zero probability to unseen words.
8. **If you have a dataset with 10,000 features, would you prefer KNN or Naive Bayes? Why?**
   *Hint:* Naive Bayes. KNN suffers terribly from the "curse of dimensionality" because in high-dimensional space, the concept of "distance" breaks down (all points become roughly equidistant). Naive Bayes handles high-dimensional sparse data (like text) exceptionally well.

### Output Interpretation
9. **You run a Naive Bayes model and look at `predict_proba`. The probabilities are very extreme (e.g., 0.99999 and 0.00001). Can you trust these numbers as true confidence intervals?**
   *Hint:* Usually not. Naive Bayes is known for being a "bad estimator" of actual probabilities because its independence assumption pushes probabilities toward the extremes. You can trust the rank ordering (the class with 0.999 is definitely more likely than the one with 0.001), but not the absolute probability value.

---

## Python Code Examples

### `example-01-basic-knn.py`

```python
"""
Example 01: Basic K-Nearest Neighbors (KNN)
Goal: Understand the mechanics of KNN and prove why scaling features is mandatory.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Create a synthetic dataset with drastically different scales
# Imagine predicting Loan Approval based on Age and Income
data = {
    'Age': [25, 30, 35, 40, 45, 50, 55, 60],
    'Income': [40000, 50000, 60000, 150000, 160000, 170000, 180000, 190000],
    'Approved': [0, 0, 0, 1, 1, 1, 1, 1]  # 0 = No, 1 = Yes
}
df = pd.DataFrame(data)

X = df[['Age', 'Income']]
y = df['Approved']

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- EXPERIMENT 1: NO SCALING ---
knn_unscaled = KNeighborsClassifier(n_neighbors=3)
knn_unscaled.fit(X_train, y_train)
pred_unscaled = knn_unscaled.predict(X_test)
acc_unscaled = accuracy_score(y_test, pred_unscaled)

print(f"Accuracy without scaling: {acc_unscaled * 100:.2f}%")
print("Why? The Income feature ranges from 40k to 190k, while Age ranges from 25 to 60.")
print("The Euclidean distance calculation completely ignores Age because the Income difference is massive.\n")

# --- EXPERIMENT 2: WITH SCALING ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_scaled = KNeighborsClassifier(n_neighbors=3)
knn_scaled.fit(X_train_scaled, y_train)
pred_scaled = knn_scaled.predict(X_test_scaled)
acc_scaled = accuracy_score(y_test, pred_scaled)

print(f"Accuracy WITH scaling: {acc_scaled * 100:.2f}%")
print("Now both Age and Income contribute equally to finding the 'nearest' neighbors.")
```

### `example-02-naive-bayes-text.py`

```python
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
```

### `example-03-real-world-fraud.py`

```python
"""
Example 03: Real-World Scenario - Tuning K in KNN for Fraud Detection
Goal: Compare KNN and Gaussian Naive Bayes on continuous data, and learn to find the best K.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# 1. Simulate a Credit Card Fraud Dataset (Continuous Data)
np.random.seed(42)
n_samples = 500

# Features: Transaction Amount ($), Distance from Home (miles)
amount = np.random.exponential(scale=100, size=n_samples)
distance = np.random.exponential(scale=50, size=n_samples)

# Make "Fraud" more likely if amount is high AND distance is high
log_odds = 0.02 * amount + 0.05 * distance - 10
prob_fraud = 1 / (1 + np.exp(-log_odds))
fraud = (np.random.rand(n_samples) < prob_fraud).astype(int)

df = pd.DataFrame({'Amount': amount, 'Distance': distance, 'Fraud': fraud})
X = df[['Amount', 'Distance']]
y = df['Fraud']

# 2. Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- EXPERIMENT 1: Gaussian Naive Bayes ---
# We use GaussianNB because our features (Amount, Distance) are continuous numbers
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)
gnb_acc = accuracy_score(y_test, gnb.predict(X_test_scaled))
print(f"Gaussian Naive Bayes Accuracy: {gnb_acc * 100:.2f}%\n")

# --- EXPERIMENT 2: Finding the best K for KNN ---
print("--- Tuning K for K-Nearest Neighbors ---")
k_values = range(1, 30, 2)  # Odd numbers from 1 to 29
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test_scaled))
    accuracies.append(acc)
    print(f"K={k:2} | Accuracy: {acc * 100:.2f}%")

best_k = k_values[np.argmax(accuracies)]
print(f"\nBest K is {best_k} with accuracy {max(accuracies)*100:.2f}%")
print("Notice how accuracy starts low (overfitting), rises, and eventually drops (underfitting).")
```
