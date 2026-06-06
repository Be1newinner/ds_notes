# Working with Large Datasets and Chunked Processing

## Lesson Overview

- This chapter covers chunked processing and memory-efficient file ingestion techniques in Pandas.
- Large datasets (e.g. 50 GB log files) exceed system RAM capacities. Attempting to load these datasets using a standard `pd.read_csv()` call will crash the system with an Out of Memory (OOM) error.
- We will cover reading datasets in chunks using the `chunksize` parameter, performing incremental aggregations, selectively loading columns using `usecols`, and utilizing high-performance file formats (like Parquet).
- Mastering chunked processing allows you to process datasets that are larger than your system's RAM.

## Learning Objectives

- Load large CSV files incrementally in memory-friendly blocks using the `chunksize` parameter.
- Implement incremental aggregation workflows (calculating totals, counts, and category frequencies) across data chunks.
- Restrict memory usage during file loads using the `usecols` parameter to ingest only necessary columns.
- Compare reading and writing performance between CSV, HDF5, and Parquet file formats.
- Understand the limits of Pandas and explore out-of-core alternatives like Dask or Polars.

---

## 1. Incremental Ingestion using `chunksize`

The `chunksize` parameter in `pd.read_csv()` returns an iterator (a `TextFileReader` object) that yields DataFrames of a specified size. This allows you to process data in blocks without loading the entire file into memory.

### Conceptual Workflow

```text
CSV File (10 GB)  ====>  pd.read_csv(..., chunksize=10000)
                              |
                              +---> Chunk 1 (10,000 rows) ---> Apply Logic / Sum
                              |
                              +---> Chunk 2 (10,000 rows) ---> Apply Logic / Sum
                              |
                              +---> Chunk 3 (10,000 rows) ---> Apply Logic / Sum
```

### Setup for Demonstration

First, let's write a mock transaction helper script to demonstrate chunked processing.

```python
import pandas as pd

# Open a chunked file reader iterator
# chunksize defines the number of rows per iteration
chunk_iterator = pd.read_csv("large_transactions.csv", chunksize=10000)

print("Iterator Type:", type(chunk_iterator))

# Fetch the first chunk
first_chunk = next(chunk_iterator)
print("Chunk Shape:", first_chunk.shape)
```

---

## 2. Incremental Aggregation across Chunks

To summarize metrics across a large file, iterate over the chunks, calculate aggregates for each block, and combine the results.

### Example: Calculating Total Revenue and Transaction Counts

```python
total_revenue = 0
total_transactions = 0

# Process the entire file in chunks
for chunk in pd.read_csv("large_transactions.csv", chunksize=20000):
    # Perform calculations on the current chunk
    total_revenue += chunk["Sales_Amt"].sum()
    total_transactions += len(chunk)

print(f"Total Combined Revenue: ${total_revenue:,.2f}")
print(f"Total Transaction Count: {total_transactions:,}")
```

---

### Example: Incremental Value Counts (Category Frequencies)

Calculating category frequencies across chunks requires updating a running summary Series.

```python
running_counts = pd.Series(dtype="int64")

for chunk in pd.read_csv("large_transactions.csv", chunksize=20000):
    # Calculate counts for the current chunk
    chunk_counts = chunk["Payment_Method"].value_counts()
    
    # Add counts to the running summary Series, filling missing categories with 0
    running_counts = running_counts.add(chunk_counts, fill_value=0)

# Convert count floats back to integers
running_counts = running_counts.astype("int64")

print("--- Consolidated Payment Method Frequencies ---")
print(running_counts)
```

---

## 3. Selective Ingestion with `usecols`

If a file contains 100 columns but your analysis only requires `CustomerID` and `Sales_Amt`, loading the entire table wastes memory. Pass a list of column names or index positions to the **`usecols`** parameter to load only the required columns.

```python
# Ingest only the necessary columns from the CSV
df_selective = pd.read_csv(
    "large_transactions.csv",
    usecols=["CustomerID", "Sales_Amt"]
)

print("--- Loaded Columns ---")
print(df_selective.columns.tolist())
```

---

## 4. High-Performance File Formats (Parquet vs HDF5 vs CSV)

CSV is a text-based format that is slow to parse and does not preserve data types. For large datasets, use binary, columnar file formats:
- **Parquet**: A columnar storage format optimized for high-performance reading and writing. It preserves data types and metadata, and supports compression.
- **HDF5**: A hierarchical data format optimized for managing large amounts of numeric data.
- **Feather**: A fast, lightweight binary format optimized for transient storage.

```python
# Write a DataFrame to Parquet format (requires pyarrow or fastparquet)
df_selective.to_parquet("transactions.parquet", compression="snappy")

# Read from Parquet format
df_parquet = pd.read_parquet("transactions.parquet")
```

---

## Common Mistakes Students Make

- **Treating the resampler as a DataFrame**: Running `df = pd.read_csv('file.csv', chunksize=10000)` returns a file reader iterator, not a DataFrame. Attempting to access columns (e.g. `df['Col']`) raises an `AttributeError`. Iterate over the object instead.
- **Loading complete columns during joins**: Performing a join on chunked datasets can cause memory issues if you load the entire lookup table into memory. Optimize joins by loading only the necessary keys.
- **Forgetting that the index resets on each chunk**: Each chunk loaded by `pd.read_csv()` starts with a zero-indexed sequence (`0` to `chunksize - 1`). If you concatenate these chunks without resetting the index, the resulting DataFrame will contain duplicate index labels.
- **Using HDF5 across multiple processes**: HDF5 files can be corrupted if accessed by multiple write processes simultaneously. Use Parquet for multi-threaded or cloud-based storage pipelines.

---

## Best Practices

- Use `usecols` to load only the necessary columns from large datasets to minimize memory usage.
- Iterate over chunks using the `chunksize` parameter when processing datasets that exceed RAM capacity.
- Standardize on the Parquet file format for storing large datasets, as it preserves data types and metadata and supports compression.
- Use incremental calculations (like summing running values) to calculate statistics across chunks without storing intermediate DataFrames in memory.

---

## Worked Real-World Examples

### Worked Example 1: Creating the Mock Large Dataset

```python
import pandas as pd
import numpy as np

# Generate a mock dataset with 100,000 rows to simulate a large CSV file
df_mock = pd.DataFrame({
    "CustomerID": np.random.randint(1000, 9999, size=100000),
    "Payment_Method": np.random.choice(["Card", "UPI", "Cash"], size=100000),
    "Sales_Amt": np.random.rand(100000) * 500,
    "Extra_Col1": np.random.rand(100000),
    "Extra_Col2": np.random.rand(100000)
})

df_mock.to_csv("large_transactions.csv", index=False)
print("Mock dataset 'large_transactions.csv' created successfully.")
```

### Output

```text
Mock dataset 'large_transactions.csv' created successfully.
```

---

### Worked Example 2: Chunked Filtering and Incremental Export

```python
import pandas as pd

# Filter out transactions where Sales_Amt > 450 in chunks
# Write the matching rows incrementally to a new CSV file
is_first = True

for chunk in pd.read_csv("large_transactions.csv", chunksize=25000):
    # Filter rows in the current chunk
    high_value = chunk[chunk["Sales_Amt"] > 450]
    
    # Write to file: write headers only for the first chunk
    high_value.to_csv(
        "high_value_transactions.csv",
        mode="w" if is_first else "a",
        header=is_first,
        index=False
    )
    is_first = False

print("High-value transactions filtered and exported successfully.")
```

### Output

```text
High-value transactions filtered and exported successfully.
```

---

## Practice Questions

1. Explain the purpose of the `chunksize` parameter in `pd.read_csv()`.
2. How does the `usecols` parameter minimize memory usage during file loads?
3. Write a command to read only columns `CustomerID` and `Payment_Method` from a CSV file.
4. What data type is returned by `pd.read_csv('file.csv', chunksize=5000)`?
5. Write a script to calculate the minimum value of a column named `Score` across a large CSV file using chunked processing.
6. Compare the read/write speeds and file sizes of CSV and Parquet file formats.
7. What are the performance risks of concatenating all chunks into a single DataFrame in memory?
8. Write a command to save a DataFrame to Parquet format using snappy compression.
9. Explain how you would calculate a global mean value across a dataset using chunked processing.
10. What out-of-core alternatives exist for processing datasets that are too large for Pandas to handle?

---

## Mini Assignments

### Assignment 1: Chunked Sales Revenue Consolidation
- Use the generated mock dataset `large_transactions.csv`.
- Read the dataset in chunks of 20,000 rows.
- Calculate the total sales revenue and average sales amount incrementally, and print the results.

### Assignment 2: Payment Method Frequency Audit
- Read `large_transactions.csv` in chunks of 15,000 rows.
- Calculate the frequency counts of `Payment_Method` incrementally across all chunks.
- Output the final consolidated counts.

### Assignment 3: Selective Ingestion and Parquet Conversion
- Load only columns `CustomerID` and `Sales_Amt` from `large_transactions.csv`.
- Save the resulting DataFrame to a binary Parquet file.
- Compare the file sizes of the CSV and Parquet files.

---

## Interview-Oriented Questions

- **What is the difference between `pd.read_csv(..., iterator=True)` and specifying `chunksize`?**
  - *Answer*: `pd.read_csv(..., chunksize=N)` returns a `TextFileReader` iterator that yields DataFrames of size `N` at each iteration step. `pd.read_csv(..., iterator=True)` returns a `TextFileReader` iterator without a fixed size, requiring you to specify the number of rows to read dynamically at each `.read(N)` call.
- **How can we calculate a global mean value across a dataset using chunked processing?**
  - *Answer*: You cannot calculate a global mean by averaging the means of each chunk, as chunk sizes may vary. Instead, keep a running sum of the values and a running count of the rows across all chunks, and divide the total sum by the total count at the end: `mean = running_sum / running_count`.
- **Why is the Parquet file format preferred over CSV for storing large datasets?**
  - *Answer*: Parquet is a binary, columnar storage format. It preserves data types and metadata (avoiding type inference during load), supports compression (reducing file sizes), and allows loading specific columns without reading the entire file, which is faster and more memory-efficient than parsing text-based CSV files.
- **What is an Out of Memory (OOM) error, and how does chunked processing prevent it?**
  - *Answer*: An OOM error occurs when a process attempts to allocate more memory than the system's available RAM. In Pandas, loading a large dataset using `pd.read_csv()` attempts to store the entire table in memory at once. Chunked processing prevents this by loading and processing the data in small, manageable blocks, keeping the memory footprint low.
- **Name three out-of-core data processing alternatives to Pandas for massive datasets.**
  - *Answer*: 1. **Dask**: Scales Pandas workflows using parallel processing. 2. **Polars**: A high-performance DataFrame library written in Rust. 3. **Apache Spark (PySpark)**: A distributed computing engine for processing massive datasets across clusters.

---

## Teaching Notes for This Chapter

- **Deconstruct Chunked Iteration**: Draw the transition from a large file to sequential DataFrame chunks on the board to illustrate chunked processing.
- **Benchmark Formats Live**: Run a live code cell in class comparing the read and write times of CSV and Parquet files to demonstrate performance differences.
- **Highlight the Mean Calculation Pitfall**: Explain why averaging chunk means is mathematically incorrect, helping students avoid this common calculation error.

---

## Chapter Wrap-up Concepts Students Must Master

- Use the `chunksize` parameter in `pd.read_csv()` to process large datasets incrementally in manageable blocks.
- Calculate statistics across chunks incrementally (e.g. keeping running sums and counts) to avoid loading the entire dataset into memory.
- Use `usecols` to load only the necessary columns from large files, reducing memory usage.
- Use binary, columnar formats (like Parquet) to store large datasets efficiently, as they preserve types and support compression.
- Out-of-core alternatives (like Dask or Polars) are preferred when datasets are too large for Pandas to handle.
