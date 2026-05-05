import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# 1. Create a simple synthetic dataset (3 dimensions)
np.random.seed(42)
data = {
    'length': np.random.normal(10, 2, 100),
    'width': np.random.normal(5, 1, 100),
    'height': np.random.normal(15, 3, 100)
}
df = pd.DataFrame(data)

print("Original Data Shape:", df.shape)
print(df.head(3))
print("-" * 30)

# 2. Initialize PCA to reduce to 2 dimensions
pca = PCA(n_components=2)

# 3. Fit and transform the data
transformed_data = pca.fit_transform(df)

print("Transformed Data Shape:", transformed_data.shape)
print(transformed_data[:3])
print("-" * 30)

# 4. Check how much information we retained
variance_ratio = pca.explained_variance_ratio_
print(f"Variance captured by PC1: {variance_ratio[0]:.2%}")
print(f"Variance captured by PC2: {variance_ratio[1]:.2%}")
print(f"Total variance captured: {np.sum(variance_ratio):.2%}")
