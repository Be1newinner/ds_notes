import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Generate data with 20 features
X, y = make_classification(n_samples=500, n_features=20, n_informative=5, random_state=42)

# 2. MANDATORY: Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Fit PCA without limiting components to see the variance spread
pca_full = PCA()
pca_full.fit(X_scaled)

# 4. Calculate cumulative variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# 5. Plot the Scree Plot (Cumulative Variance)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.axhline(y=0.90, color='r', linestyle='-', label='90% Variance Threshold')
plt.title('Explained Variance by Number of Principal Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend()
plt.grid(True)
plt.show()

# 6. Apply PCA keeping 90% of the variance
pca_90 = PCA(n_components=0.90)
X_reduced = pca_90.fit_transform(X_scaled)
print(f"Original features: {X.shape[1]}")
print(f"Reduced features (for 90% variance): {X_reduced.shape[1]}")
