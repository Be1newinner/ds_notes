import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# SIMULATING CONCEPT DRIFT
# We predict if a user clicks an ad based on 'Time_Spent_on_Site'

# 1. Training Phase (Year 2020)
# Rule: If Time > 5 mins, Click = True
X_train = np.random.uniform(1, 10, 1000).reshape(-1, 1)
y_train = (X_train > 5).astype(int).ravel()

model = LogisticRegression()
model.fit(X_train, y_train)
print(f"Initial Training Accuracy (2020): {accuracy_score(y_train, model.predict(X_train)):.2f}")

# 2. Live Deployment (Year 2021) - No Drift
# The rule remains the same.
X_live_2021 = np.random.uniform(1, 10, 500).reshape(-1, 1)
y_true_2021 = (X_live_2021 > 5).astype(int).ravel()
print(f"Live Accuracy (2021): {accuracy_score(y_true_2021, model.predict(X_live_2021)):.2f}")

# 3. Live Deployment (Year 2023) - CONCEPT DRIFT
# A competitor launched, and users are less patient.
# NEW Rule: If Time > 8 mins, Click = True. (The old rule was > 5)
# Note: The input data (X) still looks exactly the same (Uniform 1-10). Data drift tests would pass!
X_live_2023 = np.random.uniform(1, 10, 500).reshape(-1, 1)
y_true_2023 = (X_live_2023 > 8).astype(int).ravel()

# The model still predicts based on the 2020 rule (> 5)
predictions_2023 = model.predict(X_live_2023)
print(f"Live Accuracy (2023) - After Concept Drift: {accuracy_score(y_true_2023, predictions_2023):.2f}")

print("\nConclusion: The inputs didn't change, but the world changed. The model's accuracy plummeted silently.")
