import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load data
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Diabetes_Regression")

# 2. Define parameters to test
alpha_value = 0.5

print(f"Training Ridge Regression with alpha={alpha_value}...")

with mlflow.start_run(run_name="Ridge_Model"):
    
    # 3. Log the parameter manually
    mlflow.log_param("alpha", alpha_value)
    
    # Train the model
    model = Ridge(alpha=alpha_value)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # 4. Log the metrics manually
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    
    # 5. Log the actual model
    # This saves the model file so it can be deployed later
    mlflow.sklearn.log_model(model, "ridge_model_artifact")
    
    print(f"Model trained. MSE: {mse:.2f}, R2: {r2:.2f}")
    print("Run finished. Check MLflow UI to see the logged model.")
