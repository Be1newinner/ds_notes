import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Product categories dataset
data = {
    'Product': ['Laptop', 'Desktop', 'Tablet', 'Smartphone', 'Smartwatch', 'Headphones', 'Speaker'],
    'Price': [1200, 1500, 400, 800, 250, 150, 100],
    'Portability': [7, 1, 9, 10, 10, 8, 5],
    'Processing_Power': [8, 10, 5, 7, 3, 1, 1]
}
df = pd.DataFrame(data)

# Scale features
X = df[['Price', 'Portability', 'Processing_Power']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit clustering to find 3 groups
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
df['Cluster'] = model.fit_predict(X_scaled)

print(df.sort_values('Cluster'))
