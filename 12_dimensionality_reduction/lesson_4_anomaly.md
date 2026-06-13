# Anomaly and Outlier Detection

## Learning Objective
Understand what anomalies are, how dimensionality reduction and tree-based models help find them, and how to implement Isolation Forest for real-world anomaly detection.

## What Is This Topic?
Anomaly detection is the process of identifying rare items, events, or observations which raise suspicions by differing significantly from the majority of the data.

## Why This Topic Matters
Anomalies often translate to critical, actionable business insights: bank fraud, a structural defect in manufacturing, a medical problem, or an error in a data pipeline.

## Core Intuition
If you have a forest of data points, normal data points are clustered closely together, while anomalies are outcasts sitting far away. Isolation Forest algorithm works by drawing random splits through the data. Because anomalies are far away and alone, it takes very few random splits to "isolate" them. Normal points are clumped together, so it takes many splits to isolate an individual normal point.

## Key Concepts
- **Outlier vs Novelty**: 
  - *Outlier Detection*: The training data contains outliers, and the algorithm tries to fit the regions of the most concentrated training data, ignoring the deviant observations.
  - *Novelty Detection*: The training data is clean (no outliers), and the algorithm decides if a *new* observation is an anomaly.
- **Contamination**: The expected proportion of outliers in the dataset.

## Important Parameters / Options / Settings (Isolation Forest)
- `contamination`: e.g., `0.05`. Tells the algorithm to flag the top 5% most anomalous points as outliers. If set to 'auto', it uses a default threshold.
- `n_estimators`: Number of trees in the forest.
- `random_state`: Seed for reproducibility.

## Output / Result Interpretation
- **Predict Method**: Isolation Forest returns `1` for normal points (inliers) and `-1` for anomalies (outliers).
- **Decision Function**: Returns anomaly scores. Lower (more negative) scores mean more anomalous. Higher scores mean more normal.

## Real-World Uses
- Finding fraudulent credit card transactions.
- Identifying failing servers in an IT network (server metrics look different from the norm).
- Detecting sensor malfunctions in IoT devices.

## Advantages
- Isolation forest is incredibly fast and scales well to high-dimensional data.
- Does not require normal data points to follow any specific statistical distribution.

## Limitations
- Highly dependent on the `contamination` parameter, which is often a guess in real life (because true anomalies are unknown).
- Explaining *why* a point was flagged by an Isolation Forest is mathematically difficult.

## Common Mistakes
- Not scaling data when using distance-based outlier detectors (like LOF). Note: Isolation Forest is tree-based, so scaling is less critical for it, but good practice.
- Treating Anomaly Detection as a pure classification problem when you don't actually have labeled data.

## Related Methods
- **Local Outlier Factor (LOF)**: Uses density. A point is an outlier if its local density is much lower than its neighbors.
- **One-Class SVM**: Good for novelty detection on clean data.
- **PCA for Anomalies**: Points that have very high reconstruction error when passed through PCA are often anomalies.

## Code References
- `code/example-01-basic.py`
- `code/example-02-intermediate.py`
- `code/example-03-real-world.py`


---

## Isolation Forest Method and Options

### Scikit-Learn: `sklearn.ensemble.IsolationForest`

#### Purpose
To isolate observations by randomly selecting a feature and then randomly selecting a split value. The number of splittings required to isolate a sample is equivalent to the path length from the root node to the terminating node. Anomalies have shorter paths.

#### Syntax
```python
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
```

#### Common Arguments
- `n_estimators` (int): Number of trees (default=100).
- `contamination` (float or 'auto'): The proportion of outliers in the dataset. Defines the threshold on the decision function.
- `random_state` (int): Reproducibility.

#### Common Methods
- `fit(X)`: Fit the model.
- `predict(X)`: Predict if a particular sample is an outlier or not. Returns `1` for inliers, `-1` for outliers.
- `decision_function(X)`: Average anomaly score. The lower, the more abnormal.

#### Typical Workflow
1. **Preprocess**: Scale your features (e.g., using `StandardScaler`) if features have vastly different ranges.
2. **Initialize**: Create `IsolationForest` with an estimated `contamination` rate.
3. **Fit**: Call `fit()` on your dataset.
4. **Predict**: Call `predict()` to get the labels (`1` or `-1`).
5. **Filter**: Examine the rows flagged as `-1` to understand why they were flagged.

#### Common Mistakes
- **Confusing output labels**: Remember that `1` is GOOD/NORMAL, and `-1` is BAD/ANOMALY. This often trips up beginners.

---

## Anomaly Detection Code Examples Overview

Here are the code examples provided in the `code/` folder:

### 1. `code/example-01-basic.py`
Introduces Isolation Forest using synthetic 2D data. Generates a normal cluster of points and injects a few distinct outliers. Shows how to fit the model and plot the results, highlighting the anomalies in red.

### 2. `code/example-02-intermediate.py`
Focuses on the `decision_function`. Shows how to plot the contour lines of the anomaly scores, visualizing the "boundaries" the algorithm creates around normal data.

### 3. `code/example-03-real-world.py`
Simulates a real-world server metric dataset (CPU usage and Memory usage). Uses Isolation Forest to flag time periods where the server was behaving anomalously, mimicking an IT monitoring use case.

### 4. `code/example-04-advanced.py`
Uses a multi-featured CSV dataset (`server_metrics_anomaly_data.csv`) simulating complex server metrics (CPU, Memory, Disk, Network, Temp). Demonstrates preprocessing with `StandardScaler` and using `IsolationForest` on higher-dimensional data. Highlights how anomalies often appear as extreme values across multiple sensors.

---

## Anomaly Detection Practice Tasks

### Task 1: Contamination Tuning
Load the Boston Housing dataset or generate a synthetic dataset. Fit an Isolation Forest with `contamination=0.01` and another with `contamination=0.10`. How many rows are flagged as anomalies in each case?

### Task 2: LOF vs Isolation Forest
Scikit-learn also has `LocalOutlierFactor`. Apply both `IsolationForest` and `LocalOutlierFactor` to a synthetic dataset containing clusters of different densities. Compare which points each algorithm flags.

### Task 3: Salary Outliers
Create a simple dataframe of employee salaries where 98% earn between $40k and $80k, and 2% earn over $500k. Use Isolation Forest to flag the CEOs/Executives.

---

## Anomaly Detection Interview Questions

1. **Beginner**: What is the difference between an inlier and an outlier in scikit-learn's terminology?
2. **Conceptual**: How does Isolation Forest conceptually isolate an anomaly faster than a normal point?
3. **Practical**: You don't know the exact percentage of fraud in your dataset. How do you set the `contamination` parameter?
4. **Comparison**: Why might you use Isolation Forest instead of simply looking at Z-scores (standard deviations from the mean)? (Hint: Z-scores only look at one feature at a time; IsoForest looks at multi-dimensional relationships).
5. **Output**: If `model.predict()` returns an array containing `[1, 1, -1, 1]`, what does the `-1` represent?

---

## Python Code Examples

### `example-01-basic.py`

```python
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
```

### `example-02-intermediate.py`

```python
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
```

### `example-03-real-world.py`

```python
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
```

### `example-04-advanced.py`

```python
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
```
