import numpy as np

# Example 01: Rating Prediction Metrics
# Calculating RMSE and MAE manually to understand how they work.

print("--- Rating Prediction Metrics ---")

# 1. Mock Data: True Ratings vs Model Predicted Ratings
# Imagine a user rated 5 movies.
true_ratings = np.array([5.0, 4.0, 2.0, 1.0, 3.0])
predictions = np.array([4.0, 4.5, 3.0, 1.0, 4.5])

print(f"True Ratings:      {true_ratings}")
print(f"Predicted Ratings: {predictions}\n")

# 2. Calculate Mean Absolute Error (MAE)
# MAE is the simple average of the absolute errors.
absolute_errors = np.abs(true_ratings - predictions)
mae = np.mean(absolute_errors)

print("Absolute Errors for each movie:")
print(absolute_errors)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print("Meaning: On average, our predictions are off by 0.7 stars.\n")

# 3. Calculate Root Mean Squared Error (RMSE)
# RMSE squares the errors before averaging them, which punishes large mistakes heavily.
squared_errors = np.square(true_ratings - predictions)
mse = np.mean(squared_errors)
rmse = np.sqrt(mse)

print("Squared Errors for each movie:")
print(squared_errors)
# Notice how the last movie (off by 1.5 stars) generated a squared error of 2.25!
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Conclusion: RMSE will always be larger than or equal to MAE. 
# Recommender competitions (like Netflix) usually use RMSE to heavily penalize wild guesses.
