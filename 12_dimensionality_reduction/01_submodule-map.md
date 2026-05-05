# Submodule Map: Dimensionality Reduction

## 1. Linear Dimensionality Reduction (PCA)
- **Why it is taught**: It is the foundational, most widely used technique for reducing features and removing multicollinearity.
- **Theory vs Code**: Balanced. Needs strong visual intuition (projecting points onto lines/planes) followed by code.
- **Business Example**: Reducing 50 customer demographics columns into 5 principal components for a clustering model.

## 2. Non-linear Techniques (t-SNE)
- **Why it is taught**: PCA often fails to capture non-linear relationships. t-SNE is the industry standard for 2D/3D visualization of complex clusters.
- **Theory vs Code**: Mostly Code/Visual. The math is complex, so focus on intuition (preserving local neighborhoods) and visualization.
- **Business Example**: Visualizing how different document embeddings or customer segments group together organically.

## 3. Anomaly and Outlier Detection
- **Why it is taught**: Real data is messy. Detecting anomalies is critical for data cleaning and for specific use cases like fraud detection.
- **Theory vs Code**: Practical coding and intuition.
- **Business Example**: Finding fraudulent transactions in a massive dataset using Isolation Forest.

## Recommended Order of Teaching
1. PCA (Intuition -> Math concepts -> Scikit-learn implementation).
2. t-SNE (Why PCA fails -> t-SNE intuition -> Visualization).
3. Anomaly Detection (Defining outliers -> Isolation Forest / LOF -> Practical implementation).
