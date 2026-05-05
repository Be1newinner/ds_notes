"""
Example 02: K-Fold Cross Validation
This script shows how to get a more reliable performance score by splitting data multiple times.
"""

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# 1. Load a standard dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# 2. Initialize the model
model = LinearRegression()

# 3. Perform 5-Fold Cross Validation
# Instead of splitting once, we split 5 times.
# scoring='r2' means we are checking the R-squared score (closer to 1.0 is better)
print("Running 5-Fold Cross Validation...")
scores = cross_val_score(model, X, y, cv=5, scoring='r2')

# 4. Analyze the results
print(f"Individual Scores for each fold: {np.round(scores, 3)}")

print("-" * 30)
print(f"Average (Reliable) Score: {scores.mean():.3f}")
print(f"Standard Deviation (Stability): {scores.std():.3f}")

# Note:
# Look at the individual scores. One of them might be 0.58, another might be 0.42.
# If we only did a single train_test_split, we might have accidentally gotten the "lucky" 0.58 split
# and thought our model was better than it really is.
# The Average Score (mean) gives us the honest truth.
