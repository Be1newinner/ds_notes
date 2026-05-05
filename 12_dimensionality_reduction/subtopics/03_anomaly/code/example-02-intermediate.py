import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_moons

# 1. Generate crescent moon data and add uniform noise
X_moons, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
X_noise = np.random.uniform(low=-1.5, high=2.5, size=(50, 2))
X = np.vstack([X_moons, X_noise])

# 2. Fit Isolation Forest
iso = IsolationForest(contamination=0.10, random_state=42)
iso.fit(X)

# 3. Create a meshgrid to plot the decision boundary
xx, yy = np.meshgrid(np.linspace(-2, 3, 100), np.linspace(-1.5, 2, 100))
Z = iso.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 4. Plot
plt.figure(figsize=(10, 6))
# Plot contour lines of anomaly scores
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r, alpha=0.6)
plt.colorbar(label='Anomaly Score (Lower is more anomalous)')

# Plot data points
preds = iso.predict(X)
plt.scatter(X[preds == 1, 0], X[preds == 1, 1], c='white', edgecolor='k', label='Normal')
plt.scatter(X[preds == -1, 0], X[preds == -1, 1], c='red', edgecolor='k', label='Anomaly')

plt.title("Isolation Forest Decision Boundaries")
plt.legend()
plt.show()
