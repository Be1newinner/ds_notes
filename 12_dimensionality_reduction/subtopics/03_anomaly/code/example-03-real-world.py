import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 1. Simulate server metrics data (CPU and Memory) over time
np.random.seed(42)
times = pd.date_range("2023-01-01", periods=500, freq="H")

# Normal server behavior
cpu_usage = np.random.normal(40, 5, 500)
memory_usage = np.random.normal(60, 5, 500)

# Inject anomalies (Server spikes or crashes)
cpu_usage[50:55] = np.random.normal(95, 2, 5)  # CPU Spike
memory_usage[200:205] = np.random.normal(98, 1, 5) # Memory Leak
cpu_usage[400:405] = np.random.normal(5, 1, 5)   # System crash

df = pd.DataFrame({'cpu': cpu_usage, 'memory': memory_usage}, index=times)

# 2. Apply Isolation Forest to find server anomalies
iso = IsolationForest(contamination=0.03, random_state=42)
df['anomaly'] = iso.fit_predict(df[['cpu', 'memory']])

# 3. Plot the timeline
plt.figure(figsize=(15, 6))
plt.plot(df.index, df['cpu'], label='CPU Usage', alpha=0.6, color='blue')
plt.plot(df.index, df['memory'], label='Memory Usage', alpha=0.6, color='green')

# Highlight anomalies in red dots
anomalies = df[df['anomaly'] == -1]
plt.scatter(anomalies.index, anomalies['cpu'], color='red', s=50, zorder=5, label='Anomaly Detected')
plt.scatter(anomalies.index, anomalies['memory'], color='red', s=50, zorder=5)

plt.title("IT Server Monitoring - Anomaly Detection")
plt.xlabel("Time")
plt.ylabel("Usage %")
plt.legend()
plt.grid(True)
plt.show()

print("Detected Anomalous Timestamps:")
print(anomalies.head())
