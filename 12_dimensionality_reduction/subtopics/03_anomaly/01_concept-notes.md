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
