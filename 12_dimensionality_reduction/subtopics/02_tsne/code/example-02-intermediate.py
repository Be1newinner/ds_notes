import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.manifold import TSNE

# 1. Generate non-linear data (concentric circles)
# We add some noise and dimensions to make it interesting
X, y = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)

# 2. Test different perplexity values
perplexities = [5, 30, 100]

plt.figure(figsize=(15, 5))

for i, perplexity in enumerate(perplexities, 1):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    plt.subplot(1, 3, i)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='coolwarm', alpha=0.7)
    plt.title(f't-SNE (Perplexity = {perplexity})')
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()
