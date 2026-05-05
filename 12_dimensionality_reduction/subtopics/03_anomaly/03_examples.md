# Anomaly Detection Code Examples Overview

Here are the code examples provided in the `code/` folder:

## 1. `code/example-01-basic.py`
Introduces Isolation Forest using synthetic 2D data. Generates a normal cluster of points and injects a few distinct outliers. Shows how to fit the model and plot the results, highlighting the anomalies in red.

## 2. `code/example-02-intermediate.py`
Focuses on the `decision_function`. Shows how to plot the contour lines of the anomaly scores, visualizing the "boundaries" the algorithm creates around normal data.

## 3. `code/example-03-real-world.py`
Simulates a real-world server metric dataset (CPU usage and Memory usage). Uses Isolation Forest to flag time periods where the server was behaving anomalously, mimicking an IT monitoring use case.

## 4. `code/example-04-advanced.py`
Uses a multi-featured CSV dataset (`server_metrics_anomaly_data.csv`) simulating complex server metrics (CPU, Memory, Disk, Network, Temp). Demonstrates preprocessing with `StandardScaler` and using `IsolationForest` on higher-dimensional data. Highlights how anomalies often appear as extreme values across multiple sensors.
