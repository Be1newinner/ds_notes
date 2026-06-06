# MultiIndex and Hierarchical Indexing

## Lesson Overview

- This chapter covers hierarchical indexing in Pandas. We explore creating MultiIndexes, slicing MultiIndexed DataFrames using `pd.IndexSlice`, swapping and sorting index levels, and aggregating metrics by specific levels.
- Datasets often contain nested, multi-dimensional structures (e.g. tracking sales by region and product type, or recording hourly temperatures across multiple sensor locations). A MultiIndex allows you to store and query these higher-dimensional structures in a standard 2D table layout.
- We will cover constructing MultiIndexes using `from_tuples` or `from_product`, slicing levels using indexers, reorganizing levels using `.swaplevel()`, and calculating level-wise statistics.
- Mastering hierarchical indexing allows you to manage and analyze complex multi-dimensional datasets.

## Learning Objectives

- Construct MultiIndexes using `pd.MultiIndex` factory methods (`from_tuples`, `from_product`, `from_arrays`).
- Query and slice hierarchical DataFrames using `.loc[]` with tuples and `pd.IndexSlice`.
- Reorganize index hierarchies using `.swaplevel()` and sort them using `.sort_index()`.
- Aggregate metrics along specific hierarchical levels using `.groupby(level=...)`.
- Flatten MultiIndexed columns or row indexes to simplify data exports.

---

## Creating MultiIndexes in Pandas

A MultiIndex (hierarchical index) allows you to represent multiple nested dimensions in the row or column index of a DataFrame.

### Setup for Demonstration

You can create a MultiIndex using factory methods:
- `pd.MultiIndex.from_tuples()`: Creates a MultiIndex from a list of tuples.
- `pd.MultiIndex.from_product()`: Creates a MultiIndex from the Cartesian product of multiple lists.
- `pd.MultiIndex.from_arrays()`: Creates a MultiIndex from a list of arrays.

```python
import pandas as pd

# Define list of tuples representing Region and Year
index_tuples = [
    ("East", 2025), ("East", 2026),
    ("West", 2025), ("West", 2026)
]

# Create MultiIndex from tuples
multi_idx = pd.MultiIndex.from_tuples(index_tuples, names=["Region", "Year"])

# Create DataFrame with the MultiIndex
df_sales = pd.DataFrame(
    {"Sales": [12000, 15000, 8000, 9500], "Profit": [2000, 2500, 1200, 1500]},
    index=multi_idx
)

print("--- Hierarchical Sales DataFrame ---")
print(df_sales)
```

### Output

```text
--- Hierarchical Sales DataFrame ---
             Sales  Profit
Region Year               
East   2025  12000    2000
       2026  15000    2500
West   2025   8000    1200
       2026   9500    1500
```

---

## 1. Querying and Slicing Hierarchical DataFrames

Querying MultiIndexed DataFrames requires explicit coordinate specifications.

### Basic Selection with Tuples

```python
# Select a single level-0 group (returns a standard DataFrame)
east_sales = df_sales.loc["East"]
print("--- East Region Sales ---")
print(east_sales)

# Select a specific nested row using a coordinate tuple
east_2026 = df_sales.loc[("East", 2026)]
print("\n--- East 2026 Performance ---")
print(east_2026)
```

### Output

```text
--- East Region Sales ---
      Sales  Profit
Year               
2025  12000    2000
2026  15000    2500

--- East 2026 Performance ---
Sales     15000
Profit     2500
Name: (East, 2026), dtype: int64
```

---

### Advanced Slicing with `pd.IndexSlice`

To slice across multiple levels (e.g. selecting all regions for the year 2026), use **`pd.IndexSlice`**. This requires the index to be sorted chronologically.

```python
# Sort the index before slicing
df_sales = df_sales.sort_index()

# Use pd.IndexSlice
# Format: df.loc[idx[level0_slice, level1_slice], column_slice]
idx = pd.IndexSlice
sales_2026 = df_sales.loc[idx[:, 2026], "Sales"]

print("--- Sales for Year 2026 ---")
print(sales_2026)
```

### Output

```text
--- Sales for Year 2026 ---
Region  Year
East    2026    15000
West    2026     9500
Name: Sales, dtype: int64
```

---

## 2. Swapping and Sorting Index Levels

Use `.swaplevel()` to change the order of index levels. Always follow `.swaplevel()` with `.sort_index()` to ensure correct slicing.

```python
# Swap Year to level 0 and Region to level 1
df_swapped = df_sales.swaplevel("Region", "Year")

print("--- Swapped Levels (Unsorted) ---")
print(df_swapped)

# Sort index to restore structure
df_sorted = df_swapped.sort_index()
print("\n--- Swapped Levels (Sorted) ---")
print(df_sorted)
```

### Output

```text
--- Swapped Levels (Unsorted) ---
             Sales  Profit
Year Region               
2025 East    12000    2000
2026 East    15000    2500
2025 West     8000    1200
2026 West     9500    1500

--- Swapped Levels (Sorted) ---
             Sales  Profit
Year Region               
2025 East    12000    2000
     West     8000    1200
2026 East    15000    2500
     West     9500    1500
```

---

## 3. Aggregating Metrics by Index Level

Use the `level` parameter in `.groupby()` to calculate summary statistics for a specific level of the MultiIndex.

```python
# Calculate total sales by Year (aggregating across Regions)
year_totals = df_sales.groupby(level="Year")[["Sales", "Profit"]].sum()

print("--- Annual Sales Totals ---")
print(year_totals)
```

### Output

```text
--- Annual Sales Totals ---
      Sales  Profit
Year               
2025  20000    3200
2026  24500    4000
```

---

## Common Mistakes Students Make

- **Slicing unsorted MultiIndexes**: Attempting to slice an unsorted MultiIndex using `pd.IndexSlice` raises a `UnsortedIndexError`. Always sort the index first: `df = df.sort_index()`.
- **Forgetting parentheses in `.loc` tuples**: Writing `df.loc['East', 2025]` instead of `df.loc[('East', 2025)]` can cause dimension parsing errors because Pandas interprets the arguments as selecting row `'East'` and column `2025`.
- **Confusing swaplevel outcomes**: Swapping levels (`.swaplevel()`) changes the hierarchy of the index labels but does not sort them. Slicing on a swapped, unsorted index will throw errors. Always chain `.sort_index()` after swapping levels.
- **Accidental column/index loss during reset**: Running `.reset_index()` on a MultiIndexed DataFrame converts all index levels into regular columns. If you want to convert only specific levels to columns, pass the level name: `df.reset_index(level='Year')`.

---

## Best Practices

- Always sort the MultiIndex using `.sort_index()` immediately after creation to enable fast index slicing.
- Use `pd.IndexSlice` when writing complex slices across multiple hierarchical levels.
- Specify level names (rather than integer positions like `0` or `1`) to keep code readable and maintainable.
- Use `.groupby(level='LevelName')` to calculate summary statistics for specific hierarchical levels.

---

## Worked Real-World Examples

### Worked Example 1: Multi-Location Sensor Logs

```python
import pandas as pd

# Sensor logs from different cities and stations
index = pd.MultiIndex.from_product(
    [["Mumbai", "Delhi"], ["Station_A", "Station_B"]],
    names=["City", "Station"]
)

telemetry = pd.DataFrame(
    {"Temp": [32.5, 34.0, 38.2, 39.5], "Humidity": [65, 60, 45, 40]},
    index=index
)

# 1. Sort the index
telemetry = telemetry.sort_index()

# 2. Extract logs only for Station_A across all cities
idx = pd.IndexSlice
station_a_logs = telemetry.loc[idx[:, "Station_A"], :]

print("--- Station A Telemetry ---")
print(station_a_logs)
```

### Output

```text
--- Station A Telemetry ---
                  Temp  Humidity
City   Station                  
Delhi  Station_A  38.2        45
Mumbai Station_A  32.5        65
```

---

### Worked Example 2: Flat Roster Generation from MultiIndex

```python
import pandas as pd

# MultiIndexed store sales
idx = pd.MultiIndex.from_tuples(
    [("S1", "Apparel"), ("S1", "Tech"), ("S2", "Apparel")],
    names=["Store", "Category"]
)
df_stock = pd.DataFrame({"Units": [150, 420, 310]}, index=idx)

# Flatten the MultiIndex to create a standard flat DataFrame
df_flat = df_stock.reset_index()

print("--- Flattened Sales Table ---")
print(df_flat)
```

### Output

```text
--- Flattened Sales Table ---
  Store Category  Units
0    S1  Apparel    150
1    S1     Tech    420
2    S2  Apparel    310
```

---

### Worked Example 3: Hierarchical Column Transformations

```python
import pandas as pd

# MultiIndexed columns DataFrame
cols = pd.MultiIndex.from_product([["Sales", "Profit"], ["Q1", "Q2"]])
df_store = pd.DataFrame(
    [[12000, 14000, 2000, 2500]],
    index=["Store_A"],
    columns=cols
)

# Slice Q1 columns across both Sales and Profit
idx = pd.IndexSlice
q1_metrics = df_store.loc[:, idx[:, "Q1"]]

print("--- Q1 Metrics (Hierarchical Columns) ---")
print(q1_metrics)
```

### Output

```text
--- Q1 Metrics (Hierarchical Columns) ---
        Sales Profit
           Q1     Q1
Store_A  12000   2000
```

---

## Practice Questions

1. Identify the three factory methods used to construct a `pd.MultiIndex` in Pandas.
2. Write a command to sort a hierarchical DataFrame index along level 1.
3. Why does slicing a MultiIndexed DataFrame using `pd.IndexSlice` raise an error if the index is unsorted?
4. Write a command to select row `('East', 2026)` from a hierarchical DataFrame.
5. Explain how `.swaplevel()` differs from `.reorder_levels()`.
6. Write a command to calculate the mean values of a hierarchical DataFrame grouped by index level `'Region'`.
7. How can you flatten a MultiIndexed column header layout into a single flat column name array?
8. Write a script that resets only the `'Year'` level of a MultiIndex, leaving the other levels as the index.
9. Compare the indexing speed of query operations on a MultiIndex versus a flattened index.
10. Describe how to construct a MultiIndex from the columns of a DataFrame using `.set_index()`.

---

## Mini Assignments

### Assignment 1: Regional Store Inventory MultiIndex
- Create a retail store inventory DataFrame where rows are indexed hierarchically by `State` and `City`.
- Sort the index.
- Use `pd.IndexSlice` to extract the units in stock for all cities in the state of `"Maharashtra"`.

### Assignment 2: Time Series Multi-Node Slicing
- Create a telemetry log where rows are indexed by `Sensor_ID` and `Timestamp`.
- Group the logs by `Sensor_ID` (level 0) and calculate the average temperature.
- Swap the levels so that `Timestamp` is at level 0, sort the index, and slice the logs for a specific 1-hour window.

### Assignment 3: Flattening Nested Financial Columns
- Create a DataFrame with MultiIndexed column headers representing `Metric` (`"Sales"`, `"Expenses"`) and `Period` (`"2025"`, `"2026"`).
- Flatten the column headers by joining the levels with an underscore (e.g. `"Sales_2025"`).

---

## Interview-Oriented Questions

- **Why must a MultiIndex be sorted before using partial string slicing or `pd.IndexSlice`?**
  - *Answer*: Slicing relies on range boundaries. If the index is unsorted, Pandas cannot resolve the start and end positions of the range efficiently, raising an `UnsortedIndexError` to prevent incorrect results.
- **Explain the difference between `df.reset_index()` and `df.reset_index(level=...)`.**
  - *Answer*: `df.reset_index()` converts all index levels into regular columns, resetting the DataFrame to a default `RangeIndex`. `df.reset_index(level=...)` converts only the specified index level(s) into regular columns, leaving the remaining levels as the DataFrame index.
- **How can we swap the order of levels in a MultiIndexed DataFrame?**
  - *Answer*: Use the `.swaplevel(i, j)` method, where `i` and `j` are the names or integer positions of the levels to swap. Always call `.sort_index()` after swapping levels to ensure the index is sorted correctly.
- **What is the purpose of `pd.IndexSlice`, and how does it improve MultiIndex queries?**
  - *Answer*: `pd.IndexSlice` allows you to write clean, SQL-like range slices across multiple levels of a MultiIndex (e.g. `df.loc[idx[:, 2026], :]`), avoiding the need to write complex nested tuples and slices.
- **How does grouping by an index level differ from grouping by a DataFrame column?**
  - *Answer*: Grouping by an index level (`df.groupby(level='LevelName')`) uses the index metadata directly, which is faster and avoids the memory overhead of resetting the index to access the level as a column.

---

## Teaching Notes for This Chapter

- **Visualize Index Levels**: Draw a tree diagram showing how nested index categories branch out to illustrate hierarchical indexing.
- **Demonstrate sorting issues**: Show students the `UnsortedIndexError` trace in class and explain how `.sort_index()` resolves it.
- **Emphasize index level names**: Encourage students to name their index levels explicitly during creation to keep their code readable.

---

## Chapter Wrap-up Concepts Students Must Master

- Use `pd.MultiIndex` factory methods (like `from_product`) to construct hierarchical indexes.
- Always sort the MultiIndex using `.sort_index()` immediately after creation to enable index slicing.
- Use `pd.IndexSlice` to write range slices across multiple hierarchical levels.
- Swap index levels using `.swaplevel()` and sort the index afterwards to restore structure.
- Use `.groupby(level='LevelName')` to calculate summary statistics for specific hierarchical levels.
- Flatten MultiIndexes using `.reset_index()` to prepare data for exports.
