# Example 03: Professional CF using Surprise Library (Memory-Based)
# In reality, you don't write manual math loops. You use specialized libraries.
# NOTE: To run this, you must install surprise: `pip install scikit-surprise`

try:
    from surprise import Dataset, Reader, KNNBasic
    from surprise.model_selection import train_test_split
    from surprise import accuracy
except ImportError:
    print("Please install surprise using: pip install scikit-surprise")
    exit()

print("--- KNN Collaborative Filtering with Surprise ---")

# 1. Load a built-in dataset (MovieLens 100k)
# This downloads a famous dataset of 100,000 movie ratings.
print("Loading MovieLens 100k dataset...")
data = Dataset.load_builtin('ml-100k')

# 2. Split into training and testing sets
trainset, testset = train_test_split(data, test_size=0.25)

# 3. Define the Algorithm
# We will use Item-Item Collaborative filtering using Cosine Similarity
sim_options = {
    'name': 'cosine',
    'user_based': False  # False means Item-Item. True means User-User.
}

# KNNBasic is standard memory-based nearest neighbors
algo = KNNBasic(sim_options=sim_options)

# 4. Train the algorithm on the trainset
print("Training model...")
algo.fit(trainset)

# 5. Predict on the testset and evaluate
print("Testing model...")
predictions = algo.test(testset)

# 6. Calculate RMSE (Root Mean Squared Error)
rmse = accuracy.rmse(predictions)
print(f"Model RMSE: {rmse:.4f}")

# 7. Make a specific prediction
# Predict what User '196' would rate Item '302'
uid = str(196)  # User id (as string because ml-100k uses strings)
iid = str(302)  # Item id
pred = algo.predict(uid, iid)

print("\nSpecific Prediction:")
print(f"User {uid} predicting Item {iid}")
print(f"Estimated rating: {pred.est:.2f}")
