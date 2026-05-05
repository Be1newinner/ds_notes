import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Network traffic data
data = {
    'Bytes_Sent': [500, 600, 550, 480, 520, 10000, 490, 510, 530, 2],
    'Packets_Sent': [10, 12, 11, 9, 10, 200, 9, 10, 11, 1]
}
df = pd.DataFrame(data)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# We want to identify the massive spike (10000) and tiny drop (2) as anomalies
dbscan = DBSCAN(eps=0.5, min_samples=3)
df['Cluster'] = dbscan.fit_predict(X_scaled)

print("Network Traffic Analysis:")
print(df)
print("\nOutliers detected (Cluster == -1):")
print(df[df['Cluster'] == -1])
