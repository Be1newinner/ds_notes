import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load Data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

mlflow.set_experiment("Cancer_Classification_Autolog")

# 2. Enable Autologging (The Magic Feature)
# This will automatically log parameters, metrics, and models for scikit-learn
mlflow.sklearn.autolog()

# Let's run a simple loop to try different hyperparameter combinations
n_estimators_list = [10, 50, 100]
max_depth_list = [3, None]

print("Starting Autologging hyperparameter search...")

for n_est in n_estimators_list:
    for depth in max_depth_list:
        
        # We still need to start the run to separate each iteration in the UI
        with mlflow.start_run(run_name=f"RF_{n_est}_depth_{depth}"):
            print(f"Training with n_estimators={n_est}, max_depth={depth}")
            
            # Autologging handles logging the parameters automatically when we initialize
            model = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42)
            
            # Autologging handles logging the metrics and the model when we fit
            model.fit(X_train, y_train)
            
            # Note: Autologging logs training metrics. 
            # If we want test metrics, we can still log them manually alongside autolog.
            preds = model.predict(X_test)
            test_acc = accuracy_score(y_test, preds)
            mlflow.log_metric("test_accuracy", test_acc)

print("Search complete. Open 'mlflow ui' to compare all runs easily!")
