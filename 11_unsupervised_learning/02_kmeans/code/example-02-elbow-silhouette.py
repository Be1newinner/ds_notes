from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=500, centers=5, cluster_std=0.8, random_state=42)

inertia = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

fig, ax1 = plt.subplots()

ax1.plot(K_range, inertia, 'bo-')
ax1.set_xlabel('Number of clusters (K)')
ax1.set_ylabel('Inertia', color='b')

ax2 = ax1.twinx()
ax2.plot(K_range, silhouette_scores, 'rs-')
ax2.set_ylabel('Silhouette Score', color='r')

plt.title("Elbow Method & Silhouette Score")
plt.show()
