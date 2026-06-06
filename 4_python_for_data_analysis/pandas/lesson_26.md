# Performance Optimization and Memory Efficiency

## Lesson Overview

- This chapter covers performance optimization and memory efficiency in Pandas. We explore memory audits, downcasting numeric types, categorical representation, efficient loops, and vectorized calculations.
- As datasets grow to millions of rows, memory overhead and slow execution times can cause scripts to run out of memory or take hours to complete.
- We will cover `df.memory_usage(deep=True)` for auditing memory, downcasting numeric types, optimizing iteration using `.itertuples()`, and using `pd.eval()` for high-speed calculation.
- Mastering these optimization techniques allows you to write production-grade code that scales to large datasets.

## Learning Objectives

- Audit memory usage down to the byte level using `df.memory_usage(deep=True)`.
- Reduce numeric memory usage by downcasting integers and floats to smaller bit-widths.
- Optimize categorical string storage using category types.
- Avoid slow row loops (`.iterrows()`) and implement efficient iteration with `.itertuples()` or vectorized operations.
- Leverage `pd.eval()` to speed up complex mathematical operations on large DataFrames.

---

## 1. Auditing Memory with `df.memory_usage()`

To optimize memory, you must first measure how much memory each column consumes.
By default, `df.info()` and `df.memory_usage()` report only memory estimates for object columns. You **must** pass `deep=True` to inspect the actual memory consumed by strings in memory.

### Setup for Demonstration

```python
import pandas as pd
import numpy as np

# Create a sample DataFrame representing a large user roster
df_users = pd.DataFrame({
    "UserID": np.random.randint(1000, 9999, size=50000),
    "Age": np.random.randint(18, 80, size=50000),
    "Score": np.random.rand(50000) * 100,
    "Region": np.random.choice(["North", "South", "East", "West"], size=50000)
})

print("--- Default Memory Usage (Bytes) ---")
print(df_users.memory_usage(deep=True))
```

### Output

```text
--- Default Memory Usage (Bytes) ---
Index                128
UserID            400000
Age               400000
Score             400000
Region           3100000
dtype: int64
```
*Notice that the string column 'Region' consumes significantly more memory (3.1 MB) than the numeric columns.*

---

## 2. Downcasting Numeric Types

By default, Pandas assigns 64-bit types (`int64`, `float64`) to numeric columns. If your data fits within smaller ranges (e.g. `Age` values are all under 100, fitting inside `int8`), you can downcast these columns using `pd.to_numeric()`.

| Numeric Type | Range |
| :--- | :--- |
| **`int8`** | -128 to 127 |
| **`int16`** | -32,768 to 32,767 |
| **`int32`** | -2,147,483,648 to 2,147,483,647 |
| **`float32`** | 7 decimal digits of precision |

```python
# Downcast integers to the smallest possible type
df_users["Age"] = pd.to_numeric(df_users["Age"], downcast="integer")
df_users["UserID"] = pd.to_numeric(df_users["UserID"], downcast="integer")

# Downcast floats
df_users["Score"] = pd.to_numeric(df_users["Score"], downcast="float")

print("--- Optimized Numeric Types ---")
print(df_users[["UserID", "Age", "Score"]].dtypes)
print(df_users[["UserID", "Age", "Score"]].memory_usage(deep=True))
```

### Output

```text
--- Optimized Numeric Types ---
UserID     int16
Age         int8
Score    float32
dtype: object
Index         128
UserID     100000
Age         50000
Score      200000
dtype: int64
```
*Note: Memory usage for UserID dropped from 400KB to 100KB, and Age dropped from 400KB to 50KB.*

---

## 3. Optimizing Iteration: `.itertuples()` vs `.iterrows()`

If you must iterate over rows (e.g. when integrating with third-party APIs that do not support vectorization), avoid `.iterrows()`.
- **`.iterrows()`**: Slow because it yields each row as a Pandas Series, creating significant object overhead.
- **`.itertuples()`**: Much faster because it yields each row as a Python `namedtuple`, which has minimal overhead.

```python
import time

# Measure speed of .iterrows()
start = time.time()
for index, row in df_users.head(10000).iterrows():
    val = row["Age"] + row["Score"]
time_iterrows = time.time() - start

# Measure speed of .itertuples()
start = time.time()
for row in df_users.head(10000).itertuples():
    val = row.Age + row.Score
time_itertuples = time.time() - start

print(f".iterrows() Loop Time: {time_iterrows:.5f} seconds")
print(f".itertuples() Loop Time: {time_itertuples:.5f} seconds")
print(f"Speedup: {time_iterrows / time_itertuples:.1f}x faster")
```

### Output

```text
.iterrows() Loop Time: 0.35000 seconds
.itertuples() Loop Time: 0.00800 seconds
Speedup: 43.8x faster
```

---

## 4. High-Speed Expression Evaluation with `pd.eval()`

`pd.eval()` evaluates mathematical expressions on large DataFrames using the **NumExpr** engine under the hood. It compiles calculations into high-speed virtual machine instructions, avoiding the creation of large intermediate arrays in memory.

```python
# Create large DataFrame
df_large = pd.DataFrame(np.random.rand(1000000, 3), columns=["A", "B", "C"])

# Standard execution
start = time.time()
res_std = df_large["A"] + df_large["B"] * df_large["C"]
time_std = time.time() - start

# High-speed eval execution
start = time.time()
res_eval = pd.eval("df_large.A + df_large.B * df_large.C")
time_eval = time.time() - start

print(f"Standard execution time: {time_std:.5f} seconds")
print(f"pd.eval() execution time: {time_eval:.5f} seconds")
```

### Output

```text
Standard execution time: 0.01200 seconds
pd.eval() execution time: 0.00400 seconds
```
*Note: For complex calculations on DataFrames with more than 1,000,000 rows, `pd.eval()` can be significantly faster.*

---

## Common Mistakes Students Make

- **Using `.iterrows()` for simple math**: Writing a loop to add columns (`for i, r in df.iterrows(): df.loc[i, 'C'] = r['A'] + r['B']`) is extremely slow. Always use vectorized addition: `df['C'] = df['A'] + df['B']`.
- **Forgetting deep=True during memory audits**: Running `df.memory_usage()` returns memory estimates that ignore string values, leading to underestimating memory consumption. Always pass `deep=True`.
- **Casting columns to category unnecessarily**: Converting high-cardinality columns (like `Email` or `UUID`) to `category` actually increases memory usage because Pandas has to store a massive lookup table. Only use categories for columns with repeating values.
- **Accidental upcasting during arithmetic**: Adding a float32 Series to a float64 Series silently upcasts the result to float64, reversing previous downcasting optimizations. Keep types consistent.

---

## Best Practices

- Always use the `deep=True` parameter when checking memory usage to inspect string columns accurately.
- Downcast numeric columns to the smallest suitable bit-width (e.g. `int8` for ages) to reduce memory footprints.
- Convert low-cardinality string columns (like states, departments, or regions) to categorical types.
- Avoid `.iterrows()` for row iteration; use `.itertuples()` or vectorized operations instead.
- Use `pd.eval()` to compute complex mathematical expressions on large DataFrames.

---

## Worked Real-World Examples

### Worked Example 1: Optimizing a Large Transaction Ledger

```python
import pandas as pd
import numpy as np

# 1. Scaffold large transaction log
df_tx = pd.DataFrame({
    "TxID": np.random.randint(100000, 999999, size=100000),
    "Quantity": np.random.randint(1, 10, size=100000),
    "Payment_Method": np.random.choice(["Card", "UPI", "Cash"], size=100000),
    "Price": np.random.rand(100000) * 100
})

# Calculate memory before optimization
mem_before = df_tx.memory_usage(deep=True).sum()

# 2. Downcast numeric columns
df_tx["TxID"] = pd.to_numeric(df_tx["TxID"], downcast="integer")
df_tx["Quantity"] = pd.to_numeric(df_tx["Quantity"], downcast="integer")
df_tx["Price"] = pd.to_numeric(df_tx["Price"], downcast="float")

# 3. Convert Payment_Method to category
df_tx["Payment_Method"] = df_tx["Payment_Method"].astype("category")

# Calculate memory after optimization
mem_after = df_tx.memory_usage(deep=True).sum()

print("--- Memory Audit ---")
print(f"Memory Before: {mem_before / 1024**2:.2f} MB")
print(f"Memory After: {mem_after / 1024**2:.2f} MB")
print(f"Reduction: {((mem_before - mem_after) / mem_before) * 100:.2f}%")
```

### Output

```text
--- Memory Audit ---
Memory Before: 8.77 MB
Memory After: 0.95 MB
Reduction: 89.17%
```

---

### Worked Example 2: Fast Row-wise API Payload Construction

```python
import pandas as pd

df_clients = pd.DataFrame({
    "ClientID": [101, 102, 103],
    "Name": ["Aarav", "Neha", "Vikram"],
    "Email": ["aarav@test.com", "neha@test.com", "vikram@test.com"]
})

# Construct payloads using itertuples (much faster than iterrows)
payloads = []
for row in df_clients.itertuples():
    payload = {
        "id": row.ClientID,
        "recipient": row.Name,
        "contact": row.Email
    }
    payloads.append(payload)

print("--- Constructed Payloads ---")
print(payloads)
```

### Output

```text
--- Constructed Payloads ---
[{'id': 101, 'recipient': 'Aarav', 'contact': 'aarav@test.com'}, {'id': 102, 'recipient': 'Neha', 'contact': 'neha@test.com'}, {'id': 103, 'recipient': 'Vikram', 'contact': 'vikram@test.com'}]
```

---

## Practice Questions

1. Explain the importance of the `deep=True` parameter when calling `.memory_usage()`.
2. Compare the value ranges and memory sizes of `int8`, `int16`, `int32`, and `int64` data types.
3. Write a command to downcast a float column `df['Cost']` to the smallest possible float type.
4. Why is `.itertuples()` faster than `.iterrows()` for row iteration?
5. Explain how the NumExpr engine in `pd.eval()` accelerates computations compared to standard Python code.
6. Write a command to convert a string column `df['Region']` to the category data type.
7. What are the performance risks of using `.apply()` with custom Lambda functions on large DataFrames?
8. Write a script that uses `.memory_usage(deep=True)` to print the memory consumed by each column in megabytes.
9. Explain how you would optimize a DataFrame containing 10,000,000 rows where one column stores boolean flags.
10. Describe how to optimize data load times when importing CSV files with known column schemas.

---

## Mini Assignments

### Assignment 1: Transaction Ledger Memory Audit
- Create a retail transaction dataset with 100,000 rows containing columns: `OrderID` (integers), `Store_Branch` (duplicated strings), and `Sales_Amt` (floats).
- Audit memory usage before optimization.
- Downcast numeric types and convert `Store_Branch` to category.
- Recalculate and display the percentage memory reduction.

### Assignment 2: Efficient Iteration Benchmark
- Create a DataFrame containing 20,000 rows with columns `A` and `B` (floats).
- Benchmark the time taken to iterate over the rows and calculate `A * B` using `.iterrows()` versus `.itertuples()`.
- Compare the execution times and print the speedup factor.

### Assignment 3: High-Speed Mathematical Evaluation
- Create a DataFrame with 1,000,000 rows containing columns `X`, `Y`, and `Z` (floats).
- Compute `X**2 + Y**2 - Z` using standard vectorized code.
- Compute the same expression using `pd.eval()`.
- Compare the execution speeds of both methods.

---

## Interview-Oriented Questions

- **Why must we pass `deep=True` to inspect the memory usage of object columns accurately?**
  - *Answer*: By default, `df.memory_usage()` reports only the memory consumed by the references (pointers) to string objects in memory, which is a fixed size. Passing `deep=True` tells Pandas to inspect the actual memory consumed by the string values, providing an accurate memory audit.
- **Explain the performance difference between `.itertuples()` and `.iterrows()`.**
  - *Answer*: `.iterrows()` yields each row as a Pandas Series, which introduces significant object creation overhead. `.itertuples()` yields each row as a Python `namedtuple`, which has minimal overhead and preserves data types, making it much faster.
- **How does `pd.eval()` speed up computations on large DataFrames?**
  - *Answer*: `pd.eval()` parses mathematical expressions and evaluates them using the **NumExpr** engine under the hood. It compiles the calculation into high-speed virtual machine instructions, avoiding the creation of large intermediate arrays in memory, which reduces memory overhead and accelerates execution times.
- **What are the risks of downcasting integer columns to `int8`?**
  - *Answer*: `int8` can only store integer values between -128 and 127. If your column contains values outside this range, or if subsequent arithmetic operations produce values outside this range, downcasting can cause overflow errors or corrupt data values.
- **How can we optimize memory usage when loading large CSV files using `pd.read_csv()`?**
  - *Answer*: Specify column types explicitly during load time using the `dtype` parameter: `pd.read_csv('file.csv', dtype={'Age': 'int8', 'Region': 'category'})`. This prevents Pandas from default-assigning larger memory types (`int64`, `object`) during import, reducing memory usage during data load.

---

## Teaching Notes for This Chapter

- **Demonstrate Memory Audits**: Show the difference between standard memory usage reports and reports using `deep=True` in class.
- **Run Iteration Benchmarks**: Run a live code cell in class comparing the speed of `.iterrows()` and `.itertuples()` to illustrate performance differences.
- **Explain Bit-Width Ranges**: Write comparison tables showing the value ranges of different numeric data types to help students choose suitable types for downcasting.

---

## Chapter Wrap-up Concepts Students Must Master

- Pass `deep=True` to `.memory_usage()` to audit the memory usage of object columns accurately.
- Downcast numeric columns to the smallest suitable bit-width to reduce memory usage.
- Convert low-cardinality string columns to category types to optimize storage.
- Avoid `.iterrows()` for row iteration; use `.itertuples()` or vectorized operations instead.
- Use `pd.eval()` to accelerate complex mathematical calculations on large DataFrames.
