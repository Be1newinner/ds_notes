# Method Options: KNN and Naive Bayes in Scikit-Learn

## 1. K-Nearest Neighbors Classifier

### Syntax
```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
```

### Common Arguments
- **`n_neighbors`** (`int`, default=`5`): The number of neighbors to use by default for queries. This is the crucial hyperparameter to tune.
- **`weights`** (`{'uniform', 'distance'}`, default=`'uniform'`): 
  - `'uniform'`: All points in each neighborhood are weighted equally.
  - `'distance'`: Weight points by the inverse of their distance. Closer neighbors of a query point will have a greater influence than neighbors which are further away.
- **`metric`** (`str`, default=`'minkowski'`): The distance metric to use.
- **`p`** (`int`, default=`2`): Power parameter for the Minkowski metric. When `p=1`, this is equivalent to Manhattan distance. When `p=2`, this is Euclidean distance.

### Typical Workflow & Mistakes
- **Mandatory Step**: You **must** scale your features (e.g., using `StandardScaler`) before using KNN. Because it relies on physical distance calculations, a feature ranging from 0-1000 will overpower a feature ranging from 0-1.
- **Mistake**: Using KNN on a dataset with millions of rows. It has to calculate the distance to every single row for every prediction. It will be too slow.

---

## 2. Naive Bayes Classifiers

Scikit-learn provides different Naive Bayes classifiers depending on the distribution of your data.

### Gaussian Naive Bayes (For Continuous Data)
```python
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
```
- **Use Case**: When your features are continuous (e.g., height, weight, salary) and you can assume they are roughly normally distributed.
- **Arguments**: Very few arguments to tune. It's essentially a plug-and-play baseline model.

### Multinomial Naive Bayes (For Discrete/Count Data)
```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB(alpha=1.0)
```
- **Use Case**: Text classification where features represent word counts (e.g., from `CountVectorizer`).
- **Arguments**:
  - **`alpha`** (`float`, default=`1.0`): Additive (Laplace/Lidstone) smoothing parameter. Set to 0 for no smoothing. This prevents the model from predicting a probability of exactly 0 if it encounters a word in the test set that wasn't in the training set.

### Typical Workflow & Mistakes
- **Workflow for Text**: Use `CountVectorizer` or `TfidfVectorizer` to convert text to numbers, then fit `MultinomialNB`.
- **Mistake**: Using GaussianNB on sparse text data, or using MultinomialNB on continuous data containing negative values. MultinomialNB requires non-negative feature values.
