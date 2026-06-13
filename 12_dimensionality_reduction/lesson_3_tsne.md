# t-Distributed Stochastic Neighbor Embedding (t-SNE)

## Learning Objective
Understand how t-SNE differs from PCA, why it is used for visualizing high-dimensional data, and how to apply it effectively while avoiding common misinterpretations.

## What Is This Topic?
t-SNE is a non-linear dimensionality reduction technique primarily used for visualizing complex, high-dimensional datasets in 2D or 3D space.

## Why This Topic Matters
PCA often fails when data relationships are non-linear or when clusters are complex and overlapping. t-SNE excels at preserving local structures—meaning if two points are close to each other in 50-dimensional space, t-SNE works very hard to make sure they are close to each other in the 2D plot.

## Core Intuition
Imagine a crumpled piece of paper where different colored dots represent different classes. PCA tries to cast a shadow of the paper to see the dots, which might overlap them. t-SNE acts like carefully unfolding the paper and laying it flat, ensuring that dots that were physically close to each other on the paper remain close together on the table.

## Key Concepts
- **Local Structure vs Global Structure**: PCA preserves global structure (variance across the whole dataset). t-SNE preserves local structure (neighborhoods of data points).
- **Stochastic**: t-SNE is randomized. Running it twice will yield different-looking plots unless you set a random seed.
- **Perplexity**: A parameter that acts as a dial for "how many neighbors should a point consider".

## Step-by-Step Explanation
1. In the high-dimensional space, t-SNE measures the probability that a point would pick another point as its neighbor. (Close points have high probability).
2. It creates a similar probability distribution in the lower-dimensional space (2D or 3D).
3. It minimizes the difference (Kullback-Leibler divergence) between these two probability distributions using gradient descent.
4. The result is a map where local clusters are tightly preserved.

## Important Parameters / Options / Settings
- `perplexity`: Very important! Usually between 5 and 50. It balances attention between local and global aspects of your data.
- `learning_rate`: How fast the optimization runs.
- `n_iter`: Number of iterations for the optimization.

## Output / Result Interpretation
The output is purely for visualization. You cannot easily interpret the X and Y axes of a t-SNE plot. The only thing that matters is the grouping: points that form a cluster in a t-SNE plot are likely similar in the high-dimensional space. Distance between distinct clusters in t-SNE does *not* necessarily mean anything.

## Real-World Uses
- Visualizing Word Embeddings (Word2Vec) in NLP.
- Visualizing genetic sequencing data.
- Exploring image datasets (like MNIST digits) to see if the classes separate naturally.

## Advantages
- Unbeatable at visualizing non-linear relationships.
- Creates beautiful, tightly packed clusters that humans can easily interpret visually.

## Limitations
- **Computationally Expensive**: Very slow on large datasets.
- **No Transform Method**: You cannot `transform` new test data using a fitted t-SNE model. It must re-learn the whole map. It's for visualization, not feature engineering for pipelines.
- **Distances are Misleading**: The distance between two separate clusters in the plot does not accurately reflect their distance in the original data.

## Common Mistakes
- Using t-SNE to generate features for a machine learning model (use PCA or UMAP instead).
- Interpreting the distance between two clusters as meaningful.
- Not experimenting with different `perplexity` values.

## Related Methods
- **PCA**: Linear, fast, preserves global structure.
- **UMAP**: Faster than t-SNE, preserves global structure better, and allows transforming new data.

## Code References
- `code/example-01-basic.py`
- `code/example-02-intermediate.py`
- `code/example-03-real-world.py`


---

## t-SNE Method and Options

### Scikit-Learn: `sklearn.manifold.TSNE`

#### Purpose
To reduce dimensionality for the sake of visualizing high-dimensional data in 2D or 3D.

#### Syntax
```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30.0, random_state=42)
```

#### Common Arguments
- `n_components` (int): Dimension of the embedded space (usually 2 or 3).
- `perplexity` (float): Relates to the number of nearest neighbors that is used in other manifold learning algorithms. Consider selecting a value between 5 and 50. Different values can result in significantly different plots.
- `n_iter` (int): Maximum number of iterations for the optimization. Should be at least 250.
- `random_state` (int): Seed for reproducibility. Highly recommended because t-SNE is stochastic.

#### Common Methods
- `fit_transform(X)`: Fits X into an embedded space and returns that transformed output.
- **Note**: Unlike PCA, there is NO `transform(X)` method. You cannot fit t-SNE on training data and then transform test data.

#### Typical Workflow
1. Standardize the data using `StandardScaler`.
2. Initialize `TSNE` with `n_components=2` and a specific `perplexity`.
3. Use `fit_transform()` on the data.
4. Plot the resulting 2D array using matplotlib/seaborn, coloring points by their class label.

#### Common Mistakes
- **Assuming t-SNE is a preprocessing step for ML models**: Because there is no `transform` method, you cannot use it in a standard train/test pipeline. It is an Exploratory Data Analysis (EDA) tool.
- **Ignoring Perplexity**: The default perplexity might not suit your data size. You should try multiple values (e.g., 5, 30, 50).

---

## t-SNE Code Examples Overview

Here are the code examples provided in the `code/` folder to demonstrate t-SNE:

### 1. `code/example-01-basic.py`
A simple demonstration of importing and applying t-SNE to synthetic clustered data, showcasing how to visualize the results in a 2D scatter plot.

### 2. `code/example-02-intermediate.py`
Demonstrates the massive impact of the `perplexity` parameter. It loops through different perplexity values and plots the results side-by-side to show how it changes the visualization.

### 3. `code/example-03-real-world.py`
Applies both PCA and t-SNE to the famous MNIST (handwritten digits) dataset. This clearly demonstrates why t-SNE is vastly superior to PCA for visualizing complex, non-linear image data.

---

## t-SNE Practice Tasks

### Task 1: Perplexity Tuning
Generate a blobs dataset using `sklearn.datasets.make_blobs` with 500 samples, 10 features, and 4 centers.
Apply t-SNE three times using `perplexity` values of 2, 30, and 100. Plot all three side-by-side. What do you observe?

### Task 2: PCA vs t-SNE on Text
(Requires intermediate Python skills)
Load a small text dataset, apply TF-IDF vectorization. First, plot the document vectors in 2D using PCA. Then, plot them using t-SNE. Compare which visualization separates topics better.

### Task 3: 3D Visualization
Modify the basic t-SNE example to use `n_components=3`. Use `mpl_toolkits.mplot3d` to plot a 3D scatter plot of the resulting data.

---

## t-SNE Interview Questions

1. **Beginner**: What does t-SNE stand for and what is its primary use case?
2. **Comparison**: Why would you choose t-SNE over PCA for visualization?
3. **Conceptual**: Can you use t-SNE as a feature extraction step before training a Random Forest? Why or why not?
4. **Practical**: You run t-SNE and your plot looks like a single, uniform hairball with no clusters. What hyperparameter should you adjust first?
5. **Output**: If cluster A and cluster B are 10 units apart on a t-SNE plot, and cluster C is 50 units apart from A, can you conclude C is more fundamentally different from A than B is? (Answer: No, t-SNE does not preserve global distances accurately).

---

## Python Code Examples

### `example-01-basic.py`

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 1. Generate 10-dimensional clustered data
X, y = make_blobs(n_samples=300, n_features=10, centers=3, random_state=42)

# 2. Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Initialize and fit t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# 4. Plot the result
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title("Basic t-SNE Visualization")
plt.colorbar(scatter, label='Cluster Label')
plt.show()
```

### `example-02-intermediate.py`

```python
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
```

### `example-03-real-world.py`

```python
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
```
