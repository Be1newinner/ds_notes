"""
Example 01: MAE, MSE, and RMSE
This script highlights how RMSE punishes large outliers much more than MAE.
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 1. A scenario with small, normal errors
print("--- SCENARIO 1: Normal Errors ---")
y_true_1 = np.array([100, 150, 200, 250])
y_pred_1 = np.array([110, 140, 210, 240]) # Model is off by exactly 10 every time

mae_1 = mean_absolute_error(y_true_1, y_pred_1)
rmse_1 = mean_squared_error(y_true_1, y_pred_1, squared=False)

print(f"MAE:  {mae_1:.2f} (Average distance is 10)")
print(f"RMSE: {rmse_1:.2f} (Also 10, because all errors are exactly the same size)")
print()

# 2. A scenario with one MASSIVE outlier error
print("--- SCENARIO 2: One Massive Outlier ---")
y_true_2 = np.array([100, 150, 200, 250])
# Model is perfect for the first 3, but wildly wrong on the last one (off by 40)
y_pred_2 = np.array([100, 150, 200, 290]) 

mae_2 = mean_absolute_error(y_true_2, y_pred_2)
rmse_2 = mean_squared_error(y_true_2, y_pred_2, squared=False)

print(f"MAE:  {mae_2:.2f} (Total error of 40 / 4 houses = 10)")
print(f"RMSE: {rmse_2:.2f} (Much larger than MAE!)")

# Interpretation:
# Notice how both scenarios have an MAE of 10. The "total" amount of error is the same.
# But Scenario 2 has a much higher RMSE.
# This proves that RMSE heavily penalizes models that make single, massive mistakes.
