"""
Example 01: The Basic Train-Test Split
This script demonstrates how to split data and why it matters.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 1. Create a dummy dataset
# Let's say X is the square footage of a house, y is the price
np.random.seed(42)
X = np.random.rand(100, 1) * 2000 + 500  # Houses from 500 to 2500 sq ft
# Price is roughly $150 per sq ft, plus some random noise
y = X * 150 + (np.random.randn(100, 1) * 20000) 

# 2. Split the data
# We keep 20% of the data hidden for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Total records: {len(X)}")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print("-" * 30)

# 3. Train the model ONLY on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluate the model
# Let's see how well it memorized the training data
train_predictions = model.predict(X_train)
train_error = mean_absolute_error(y_train, train_predictions)

# Let's see how well it actually performs on unseen data
test_predictions = model.predict(X_test)
test_error = mean_absolute_error(y_test, test_predictions)

print(f"Error on Training Data (Memorized): ${train_error:,.2f}")
print(f"Error on Testing Data (Unseen):   ${test_error:,.2f}")

# Notice how the error is usually slightly higher on the testing data.
# This proves why we must evaluate models on unseen data!
