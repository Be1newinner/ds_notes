import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Simulate Actual vs Predicted values
y_actual = np.array([100, 150, 200, 250, 300])

# Model A is consistently off by a little bit
y_pred_A = np.array([110, 160, 210, 260, 310])

# Model B is mostly perfect, but makes one HUGE mistake
y_pred_B = np.array([100, 150, 200, 250, 350]) 

# 2. Function to print all metrics
def print_metrics(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    
    print(f"--- {model_name} ---")
    print(f"MAE:  {mae:.2f} (Average absolute error)")
    print(f"MSE:  {mse:.2f} (Average squared error - units are squared!)")
    print(f"RMSE: {rmse:.2f} (Root of MSE - penalizes large errors)")
    print(f"R2:   {r2:.4f} (Variance explained)")
    print("")

# 3. Compare the models
print_metrics("Model A (Consistent small errors)", y_actual, y_pred_A)
print_metrics("Model B (One huge error)", y_actual, y_pred_B)

print("Key Takeaway:")
print("Model A and Model B both have a MAE of 10.")
print("HOWEVER, Model B has a much worse RMSE (22.36 vs 10.00).")
print("This shows how RMSE heavily penalizes large errors (the single error of 50 in Model B).")
