import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time

# 1. Load Digits dataset (8x8 images of handwritten digits = 64 features)
digits = load_digits()
X = digits.data
y = digits.target

# 2. Apply PCA
t0 = time.time()
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
t1 = time.time()
print(f"PCA time: {t1-t0:.2f} seconds")

# 3. Apply t-SNE
t0 = time.time()
tsne = TSNE(n_components=2, perplexity=40, random_state=42)
X_tsne = tsne.fit_transform(X)
t1 = time.time()
print(f"t-SNE time: {t1-t0:.2f} seconds")

# 4. Plot comparison side-by-side
plt.figure(figsize=(16, 7))

# PCA Plot
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.title('PCA on Digits Dataset')
plt.colorbar()

# t-SNE Plot
plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.title('t-SNE on Digits Dataset')
plt.colorbar()

plt.show()
# Notice how t-SNE separates the digits beautifully compared to PCA!
