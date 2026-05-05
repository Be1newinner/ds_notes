import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 1. Generate normal data
np.random.seed(42)
X_normal = 0.3 * np.random.randn(100, 2)

# 2. Generate anomalous data (outliers)
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# 3. Combine them
X = np.vstack([X_normal, X_outliers])

# 4. Initialize and fit Isolation Forest
# We guess that about 15% of our data might be outliers
iso = IsolationForest(contamination=0.15, random_state=42)
preds = iso.fit_predict(X)

# preds array contains 1 for normal, -1 for anomaly
colors = np.array(['red' if p == -1 else 'blue' for p in preds])

# 5. Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolor='k')
plt.title("Isolation Forest Anomaly Detection\n(Blue = Normal, Red = Anomaly)")
plt.grid(True)
plt.show()
