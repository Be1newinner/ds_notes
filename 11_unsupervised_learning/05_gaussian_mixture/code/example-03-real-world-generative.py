import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

# Historical Temperature and Ice Cream Sales
data = {
    'Temp_C': [25, 26, 28, 30, 32, 10, 12, 11, 15, 14],
    'Sales':  [200, 210, 250, 300, 310, 50, 60, 55, 80, 70]
}
df = pd.DataFrame(data)

gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(df)

# Soft Clustering Probabilities
probs = gmm.predict_proba(df)
df['Prob_Cluster_0'] = np.round(probs[:, 0], 2)
df['Prob_Cluster_1'] = np.round(probs[:, 1], 2)

print("Data with Probabilities:")
print(df)

# GMMs are generative! We can generate new synthetic data days
new_data, new_labels = gmm.sample(3)
print("\nGenerated Synthetic Days (Temp, Sales):")
print(np.round(new_data, 1))
