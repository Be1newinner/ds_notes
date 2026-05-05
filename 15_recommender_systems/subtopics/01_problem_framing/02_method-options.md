# Method Options and Properties: Problem Framing

While Problem Framing is largely conceptual, there are specific Pandas and Numpy methods used to construct and manipulate the Utility Matrix.

## 1. Creating the Utility Matrix

### `pandas.DataFrame.pivot_table()`
- **Purpose**: Reshapes data from a long format (transactions/interactions) to a wide format (User-Item matrix).
- **Syntax**: `df.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0)`
- **Common Arguments**:
  - `index`: The column to use for rows (typically User ID).
  - `columns`: The column to use for columns (typically Item ID).
  - `values`: The column containing the interaction/rating.
  - `fill_value`: Value to replace missing combinations (usually 0, though in some algorithms NaN is kept).
  - `aggfunc`: Function to use if there are multiple interactions (e.g., `mean` or `sum`).
- **Return Type**: A new Pandas DataFrame.

### `scipy.sparse.csr_matrix`
- **Purpose**: When utility matrices become too large for Pandas memory, they must be converted to sparse matrices.
- **Syntax**: `csr_matrix(df.values)`
- **Common Usage**: Scikit-learn and specialized recommender libraries require CSR (Compressed Sparse Row) matrices for efficiency.

## 2. Building a Popularity Baseline

### `pandas.DataFrame.groupby()` and `agg()`
- **Purpose**: To calculate the popularity or average rating of items.
- **Syntax**: `df.groupby('item_id').agg({'rating': ['count', 'mean']})`
- **Common Workflow**:
  1. Group by item.
  2. Count the number of interactions.
  3. Calculate the average rating.
  4. Filter out items with very few interactions.
  5. Sort by rating or count descending.

### `pandas.DataFrame.sort_values()`
- **Purpose**: To rank the items to generate the top N recommendations.
- **Syntax**: `df.sort_values(by='rating_count', ascending=False)`

## Common Mistakes
- **Memory Errors with `pivot_table`**: Calling `pivot_table` on millions of rows can cause a RAM crash. In real-world scenarios, sparse matrices (`scipy.sparse`) are required.
- **Not filling NaNs**: Machine learning algorithms cannot handle `NaN` values in the utility matrix. You must decide whether to fill them with 0, the user's mean, or the item's mean.
