# Methods, Options, and Properties: MLflow

This document explains the common methods used in the `mlflow` library for experiment tracking.

## 1. Core MLflow Methods

### Setting up the Experiment
- **`mlflow.set_tracking_uri(uri)`**: Sets the destination where tracking data is saved. If not set, it saves to a local `mlruns` directory by default.
- **`mlflow.set_experiment(experiment_name)`**: Sets the active experiment. If it doesn't exist, MLflow creates it. All subsequent runs are logged under this experiment.

### Managing Runs
- **`mlflow.start_run(run_name=None)`**: Starts a new tracking run. Best practice is to use it in a `with` statement so the run automatically ends when the block finishes.
  ```python
  with mlflow.start_run(run_name="RandomForest_Trial_1"):
      # Training code here
  ```

### Logging Data
Inside the `with mlflow.start_run():` block, you use these methods:
- **`mlflow.log_param(key, value)`**: Logs a single parameter (e.g., `mlflow.log_param("max_depth", 5)`).
- **`mlflow.log_params(dict)`**: Logs multiple parameters at once from a dictionary.
- **`mlflow.log_metric(key, value)`**: Logs a single evaluation metric (e.g., `mlflow.log_metric("accuracy", 0.92)`).
- **`mlflow.log_metrics(dict)`**: Logs multiple metrics at once.
- **`mlflow.log_artifact(local_path)`**: Logs a local file or directory as an artifact (e.g., a matplotlib plot saved as a PNG).

---

## 2. Framework-Specific Logging

MLflow has built-in modules for popular frameworks to handle the complex serialization of models automatically.

### Scikit-learn (`mlflow.sklearn`)
- **`mlflow.sklearn.log_model(sk_model, artifact_path)`**: Logs a scikit-learn model. `sk_model` is the trained model object, and `artifact_path` is the folder name within the run where it will be saved.
- **`mlflow.sklearn.load_model(model_uri)`**: Loads a previously logged model so you can use it for predictions.

### PyTorch / TensorFlow
- Similar functions exist for deep learning frameworks: `mlflow.pytorch.log_model()`, `mlflow.tensorflow.log_model()`.

---

## 3. Autologging (The Magic Feature)

Instead of manually logging every parameter and metric, MLflow can automatically hook into supported libraries and log everything for you.

- **`mlflow.autolog()`**: Tries to enable autologging for all supported libraries (scikit-learn, xgboost, lightgbm, statsmodels, etc.) found in the environment.
- **`mlflow.sklearn.autolog()`**: Specifically enables autologging for scikit-learn. It automatically logs:
  - Model parameters (e.g., `n_estimators`, `criterion`).
  - Common training metrics.
  - The trained model itself.

*Note: Autologging is incredibly convenient but provides less fine-grained control over exactly what gets logged compared to manual logging.*

---

## 4. The MLflow UI

The UI is how you interact with the logged data.
- **Starting the UI**: Open your terminal in the directory where your script ran (where the `mlruns` folder is) and run the command:
  `mlflow ui`
- **Access**: Open a web browser and go to `http://127.0.0.1:5000` (or `http://localhost:5000`).
- **Features**:
  - Left panel: List of Experiments.
  - Main panel: Table of Runs for the selected experiment.
  - Click on a run to see detailed parameters, metrics, and download artifacts.
  - Check the boxes next to multiple runs and click "Compare" to view scatter plots and parallel coordinates plots of how parameters affected metrics.
