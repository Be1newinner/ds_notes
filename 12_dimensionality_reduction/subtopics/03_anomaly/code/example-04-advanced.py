import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import os

# 1. Load the generated dataset
# We use os.path to ensure it works even if run from different directories
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'server_metrics_anomaly_data.csv')

try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: {data_path} not found.")
    print("Please ensure the CSV file is in the same directory as this script.")
    exit()

# 2. Preprocessing
# We exclude 'timestamp' as it's not a numerical feature for the model
features = ['cpu_usage', 'memory_usage', 'disk_io', 'network_in', 'network_out', 'temperature']
X = df[features]

# Scaling is often good practice, though Isolation Forest is tree-based and less sensitive
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Anomaly Detection
# We assume a contamination of 2% as we injected that much
iso = IsolationForest(contamination=0.02, random_state=42)
df['anomaly_score'] = iso.fit_predict(X_scaled)

# Convert labels: 1 is normal, -1 is anomaly
df['is_anomaly'] = df['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)

# 4. Results Interpretation
anomalies = df[df['is_anomaly'] == 1]
print(f"Total records: {len(df)}")
print(f"Detected anomalies: {len(anomalies)}")

# Show a few detected anomalies
print("\nSample Detected Anomalies:")
print(anomalies[features].head())

# 5. Visualization (Plotting CPU vs Memory)
plt.figure(figsize=(10, 6))
plt.scatter(df['cpu_usage'], df['memory_usage'], 
            c=df['anomaly_score'], cmap='RdYlGn', alpha=0.6, label='Data Points')
plt.colorbar(label='Anomaly Score (Green=Normal, Red=Anomaly)')
plt.scatter(anomalies['cpu_usage'], anomalies['memory_usage'], 
            edgecolor='black', facecolor='none', s=100, label='Detected Outliers')

plt.title('Advanced Anomaly Detection: CPU vs Memory Usage')
plt.xlabel('CPU Usage (%)')
plt.ylabel('Memory Usage (%)')
plt.legend()
plt.grid(True)
plt.show()
