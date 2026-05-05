# Example 04: Matrix Factorization (Model-Based CF)
# This is the algorithm (SVD) that won the $1 Million Netflix Prize.
# NOTE: To run this, you must install surprise: `pip install scikit-surprise`

try:
    from surprise import Dataset, SVD
    from surprise.model_selection import cross_validate
except ImportError:
    print("Please install surprise using: pip install scikit-surprise")
    exit()

print("--- Matrix Factorization (SVD) with Surprise ---")

# 1. Load the dataset
data = Dataset.load_builtin('ml-100k')

# 2. Initialize the SVD algorithm
# n_factors = 50 means we are compressing the huge matrix into 50 "hidden" latent factors.
algo = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)

# 3. Evaluate using 5-Fold Cross Validation
# This automatically splits the data, trains, and tests 5 times to give a robust metric.
print("Running 5-fold cross validation...")
results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# 4. Print Average RMSE
print(f"\nAverage RMSE across 5 folds: {results['test_rmse'].mean():.4f}")

# Notice how the RMSE for SVD is usually lower (better) than the RMSE for KNNBasic
# in the previous example. Model-based approaches usually outperform memory-based approaches
# and they are MUCH faster to query in production.

# 5. Train on the whole dataset to put into "production"
trainset = data.build_full_trainset()
algo.fit(trainset)

# Predict User 196 rating Item 302
uid = str(196)
iid = str(302)
pred = algo.predict(uid, iid)

print(f"\nProduction Prediction for User {uid} on Item {iid}: {pred.est:.2f}")
