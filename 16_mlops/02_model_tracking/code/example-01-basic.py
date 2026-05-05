import mlflow
import random

# 1. Set the experiment name
# If it doesn't exist, MLflow creates it
mlflow.set_experiment("Basic_Math_Experiment")

print("Starting experiment run...")

# 2. Start a run using a 'with' block
with mlflow.start_run(run_name="Run_1_Addition"):
    
    # Define some parameters (inputs)
    param_a = 10
    param_b = 20
    operation = "add"
    
    # 3. Log the parameters
    mlflow.log_param("value_a", param_a)
    mlflow.log_param("value_b", param_b)
    mlflow.log_param("operation", operation)
    
    # Simulate some "training" or calculation
    if operation == "add":
        result = param_a + param_b
    
    # Simulate a metric with some random noise
    accuracy = result + random.uniform(-2, 2)
    
    # 4. Log the metric (output)
    mlflow.log_metric("fake_accuracy", accuracy)
    
    print(f"Run finished. Result: {result}, Logged Metric: {accuracy:.2f}")

print("To view results, open terminal in this folder and run: mlflow ui")
