# t-SNE Practice Tasks

## Task 1: Perplexity Tuning
Generate a blobs dataset using `sklearn.datasets.make_blobs` with 500 samples, 10 features, and 4 centers.
Apply t-SNE three times using `perplexity` values of 2, 30, and 100. Plot all three side-by-side. What do you observe?

## Task 2: PCA vs t-SNE on Text
(Requires intermediate Python skills)
Load a small text dataset, apply TF-IDF vectorization. First, plot the document vectors in 2D using PCA. Then, plot them using t-SNE. Compare which visualization separates topics better.

## Task 3: 3D Visualization
Modify the basic t-SNE example to use `n_components=3`. Use `mpl_toolkits.mplot3d` to plot a 3D scatter plot of the resulting data.
