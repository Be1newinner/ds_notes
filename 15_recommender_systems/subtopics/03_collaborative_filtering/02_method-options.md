# Method Options and Properties: Collaborative Filtering

Collaborative Filtering can be implemented manually using Pandas/Numpy for learning purposes, but in practice, specialized libraries like `Surprise` or `Implicit` are used.

## 1. Manual Implementation (Pandas)

### `pandas.DataFrame.corr(method='pearson')`
- **Purpose**: Calculates the Pearson correlation coefficient between columns.
- **Usage in CF**: Used in Item-Item CF to find how correlated the ratings of Item A are with Item B across all users.
- **Why Pearson?**: It automatically handles mean-centering. It accounts for users who generally rate things higher or lower than average.

## 2. The `Surprise` Library
Scikit-Learn does not have built-in support for recommender systems. The `Surprise` library (Simple Python RecommendatIon System Engine) is the standard for teaching and basic model-based CF.

### `surprise.Dataset.load_from_df()`
- **Purpose**: Loads a Pandas DataFrame into the format required by Surprise.
- **Syntax**: 
  ```python
  from surprise import Reader, Dataset
  reader = Reader(rating_scale=(1, 5))
  data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)
  ```
- **Constraint**: The DataFrame *must* have exactly 3 columns in the specific order: user, item, rating.

### `surprise.prediction_algorithms.knns.KNNBasic`
- **Purpose**: Memory-based collaborative filtering (User-User or Item-Item).
- **Options**:
  - `sim_options = {'name': 'cosine', 'user_based': True}`: Does User-User Cosine similarity.
  - `sim_options = {'name': 'pearson', 'user_based': False}`: Does Item-Item Pearson correlation.
  - `k`: The max number of neighbors to consider (default 40).

### `surprise.prediction_algorithms.matrix_factorization.SVD`
- **Purpose**: Model-based collaborative filtering using Singular Value Decomposition (Matrix Factorization).
- **Parameters**:
  - `n_factors`: Number of latent factors (default 100). Higher means more complex model, but risks overfitting.
  - `n_epochs`: Number of iterations of the SGD procedure (default 20).
  - `lr_all`: Learning rate (default 0.005).
  - `reg_all`: Regularization term to prevent overfitting (default 0.02).

## 3. Workflow for Prediction

### `model.fit(trainset)`
- **Purpose**: Trains the algorithm on the utility matrix.

### `model.predict(uid, iid)`
- **Purpose**: Predicts the rating a specific user (`uid`) would give a specific item (`iid`).
- **Return Type**: A `Prediction` object containing the estimated rating (`est`).

## Common Mistakes
- **Loading data incorrectly into Surprise**: Forgetting the `Reader` object or passing columns in the wrong order.
- **Using Memory-Based CF on huge datasets**: `KNNBasic` will run out of memory or take hours if you have millions of rows. Switch to `SVD`.
