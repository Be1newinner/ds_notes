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
