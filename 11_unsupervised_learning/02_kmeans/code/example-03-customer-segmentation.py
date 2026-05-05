import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Simulated e-commerce customer data
data = {
    'Annual_Spend': [1500, 200, 3000, 250, 1800, 4000, 150, 2800],
    'Purchase_Frequency': [12, 2, 24, 3, 15, 30, 1, 20]
}
df = pd.DataFrame(data)

# Scaling is critical!
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Fit KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
df['Cluster'] = kmeans.fit_predict(df_scaled)

print("Segmented Customers:")
print(df)

# Analyze the centers (in original scale)
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(centers_original, columns=['Annual_Spend', 'Purchase_Frequency'])
centers_df.index.name = 'Cluster'
print("\nCluster Profiles (Averages):")
print(centers_df)
