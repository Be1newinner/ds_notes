# Examples: Model Tracking with MLflow

This document outlines the practical examples provided for learning experiment tracking.

## Code References

- `code/example-01-basic.py` — **Manual Logging**: Demonstrates how to start a run and manually log simple parameters and metrics using `mlflow.log_param` and `mlflow.log_metric`.
- `code/example-02-intermediate.py` — **Logging Scikit-Learn Models**: Shows how to train a model, log its metrics, and save the actual model file using `mlflow.sklearn.log_model`.
- `code/example-03-real-world.py` — **Autologging and Hyperparameter Tuning**: A realistic example demonstrating how to use `mlflow.autolog()` in conjunction with a loop that tries different hyperparameters, automatically logging all variations.

## How to use these examples

1. Run the Python scripts sequentially from your terminal.
2. After running them, you will notice a new directory created called `mlruns`.
3. Open a new terminal in the same directory and type `mlflow ui`.
4. Open your web browser and navigate to the address provided (usually `http://127.0.0.1:5000`).
5. Explore the UI to see how the data from the scripts was recorded.
