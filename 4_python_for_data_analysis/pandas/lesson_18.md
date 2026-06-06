# Reshaping and Tidy Data (stack, unstack, melt, wide to long)

## Lesson Overview

- This chapter explores reshaping techniques in Pandas to clean and restructure datasets. We cover Tidy Data principles, stacking and unstacking indices, melting wide tables into long tables, and converting structured column headers using `pd.wide_to_long()`.
- Data is often structured for human readability (wide format) rather than computer processing (long format). Wide tables make visual scanning easy but complicate grouping, filtering, and modeling.
- We will cover methods like `df.stack()` and `df.unstack()`, explore melting using `pd.melt()`, and use `pd.wide_to_long()` to parse columns with naming patterns.
- Mastering these reshaping workflows allows you to clean messy datasets and prepare them for machine learning pipelines.

## Learning Objectives

- Apply Tidy Data principles to structure datasets for downstream analysis.
- Pivot column levels to row indices using `df.stack()`.
- Pivot row indices back to columns using `df.unstack()`.
- Melt wide-format tables into tidy, long-format tables using `pd.melt()`.
- Restructure DataFrames with systematic column patterns using `pd.wide_to_long()`.

---

## Tidy Data Principles

According to Hadley Wickham’s Tidy Data rules:
1. Each variable forms a column.
2. Each observation forms a row.
3. Each type of observational unit forms a table.

```text
Wide Format (Untidy for analysis)        Long Format (Tidy)
Product  Q1_Sales  Q2_Sales              Product  Quarter  Sales
Laptop   12000     14000      ====>      Laptop   Q1_Sales 12000
Mouse    450       600                   Laptop   Q2_Sales 14000
                                         Mouse    Q1_Sales 450
                                         Mouse    Q2_Sales 600
```

---

## 1. Stacking and Unstacking Indices

### `df.stack()`
`.stack()` pivots the innermost column level to the innermost row index, converting a DataFrame into a MultiIndexed Series.

### Setup for Demonstration

```python
import pandas as pd

# MultiIndexed columns DataFrame
columns = pd.MultiIndex.from_product([["Sales", "Profit"], ["Q1", "Q2"]])
df_financials = pd.DataFrame(
    [[12000, 14000, 2000, 2500], [450, 600, 100, 150]],
    index=["Laptop", "Mouse"],
    columns=columns
)

print("--- Original Financials Table ---")
print(df_financials)
```

### Output

```text
--- Original Financials Table ---
        Sales         Profit       
           Q1     Q2      Q1     Q2
Laptop  12000  14000    2000   2500
Mouse     450    600     100    150
```

---

### Applying Stacking

```python
# Stack the columns (Q1/Q2) into the row index
stacked_df = df_financials.stack()

print("--- Stacked DataFrame ---")
print(stacked_df)
```

### Output

```text
--- Stacked DataFrame ---
             Profit  Sales
Laptop Q1      2000  12000
       Q2      2500  14000
Mouse  Q1       100    450
       Q2       150    600
```
*Note: Stacking has shifted the inner column level ('Q1'/'Q2') into the row index, creating a MultiIndex.*

---

### `df.unstack()`
`.unstack()` reverses the operation, pivoting a row index level to a column level. By default, it unstacks the innermost level.

```python
# Unstack back to columns
unstacked_df = stacked_df.unstack(level=-1)

print("--- Unstacked (Level -1) ---")
print(unstacked_df)
```

### Output

```text
--- Unstacked (Level -1) ---
       Profit       Sales       
           Q1    Q2    Q1     Q2
Laptop   2000  2500 12000  14000
Mouse     100   150   450    600
```

---

## 2. Melting Wide Tables to Long Tables with `pd.melt()`

`pd.melt()` reshapes a DataFrame from wide format to long format:
- **`id_vars`**: Identifier columns that remain unchanged.
- **`value_vars`**: Columns to unpivot into rows.
- **`var_name`**: Name of the new column storing the old column headers.
- **`value_name`**: Name of the new column storing the cell values.

```python
# Wide-format sales table
df_wide = pd.DataFrame({
    "Product": ["Laptop", "Mouse"],
    "Q1_Sales": [12000, 450],
    "Q2_Sales": [14000, 600]
})

# Melt to long format
df_long = pd.melt(
    df_wide,
    id_vars=["Product"],
    value_vars=["Q1_Sales", "Q2_Sales"],
    var_name="Quarter",
    value_name="Sales"
)

print("--- Melted Long DataFrame ---")
print(df_long)
```

### Output

```text
--- Melted Long DataFrame ---
  Product   Quarter  Sales
0  Laptop  Q1_Sales  12000
1   Mouse  Q1_Sales    450
2  Laptop  Q2_Sales  14000
3   Mouse  Q2_Sales    600
```

---

## 3. Reshaping with `pd.wide_to_long()`

`pd.wide_to_long()` is a specialized method for converting columns with systematic naming patterns (like `Sales_2025`, `Sales_2026`) into a long format.

```python
# Wide DataFrame with year prefixes
df_years = pd.DataFrame({
    "ProductID": [101, 102],
    "Sales_2025": [12000, 450],
    "Sales_2026": [15000, 500],
    "Profit_2025": [2000, 100],
    "Profit_2026": [2500, 120]
})

# Reshape wide to long
# stubnames: prefixes of the columns to unpivot (e.g. Sales, Profit)
# i: identifier column(s)
# j: name of the new index column (e.g. Year)
# sep: separator in the column names
df_tidy = pd.wide_to_long(
    df_years,
    stubnames=["Sales", "Profit"],
    i="ProductID",
    j="Year",
    sep="_"
).reset_index()

print("--- Wide to Long Reshaped DataFrame ---")
print(df_tidy)
```

### Output

```text
--- Wide to Long Reshaped DataFrame ---
   ProductID  Year  Sales  Profit
0        101  2025  12000    2000
1        102  2025    450     100
2        101  2026  15000    2500
3        102  2026    500     120
```

---

## Common Mistakes Students Make

- **Confusing Stacking and Unstacking**: Remembering which operation is which can be challenging. Think of `.stack()` as stacking columns vertically into rows (making the table taller), and `.unstack()` as shifting rows horizontally into columns (making the table wider).
- **Losing index labels during `.melt()`**: If the columns you want to preserve as identifiers are stored in the row index, `.melt()` will discard them unless you reset the index first: `df.reset_index().melt(...)`.
- **Mismatching prefixes in `pd.wide_to_long()`**: If column names have inconsistent prefixes (like `Sales2025` and `Profit_2026`), `pd.wide_to_long()` will fail to parse them. Column names must follow a consistent separator pattern.
- **Inflated memory from duplicate row indices**: Unstacking columns with missing combinations can create a sparse DataFrame populated mostly with `NaN`s, increasing memory usage. Clean up missing values before reshaping.

---

## Best Practices

- Standardize datasets into tidy long formats before running statistics, visualizations, or machine learning pipelines.
- Use `.reset_index()` after stacking or unstacking to flatten MultiIndexes into regular columns.
- Explicitly define `var_name` and `value_name` inside `pd.melt()` to avoid columns default-naming to `'variable'` and `'value'`.
- Use `pd.wide_to_long()` for parsing datasets with columns that contain variables (like years or months) in their headers.

---

## Worked Real-World Examples

### Worked Example 1: Reshaping Classroom Test Scores

```python
import pandas as pd

# Classroom test scores in wide format
scores_wide = pd.DataFrame({
    "Student": ["Aarav", "Neha", "Vikram"],
    "Math": [88, 92, 79],
    "Science": [95, 89, 85]
})

# Melt to tidy format
scores_tidy = pd.melt(
    scores_wide,
    id_vars=["Student"],
    value_vars=["Math", "Science"],
    var_name="Subject",
    value_name="Score"
)

print("--- Tidy Classroom Scores ---")
print(scores_tidy)
```

### Output

```text
--- Tidy Classroom Scores ---
  Student  Subject  Score
0   Aarav     Math     88
1    Neha     Math     92
2  Vikram     Math     79
3   Aarav  Science     95
4    Neha  Science     89
5  Vikram  Science     85
```

---

### Worked Example 2: Normalizing Multi-Node Sensor Telemetry

```python
import pandas as pd

# Multi-sensor logs
sensor_grid = pd.DataFrame(
    [[25.4, 60.2], [26.1, 58.9]],
    index=["12:00", "13:00"],
    columns=["Node_Temp", "Node_Humidity"]
)

# Stack columns into index
sensor_series = sensor_grid.stack()

# Convert back to DataFrame with clean column names
sensor_flat = sensor_series.reset_index()
sensor_flat.columns = ["Timestamp", "Metric", "Value"]

print("--- Normalized Sensor Telemetry ---")
print(sensor_flat)
```

### Output

```text
--- Normalized Sensor Telemetry ---
  Timestamp    Metric  Value
0     12:00      Node   25.4 # split by underscore was not requested, so stack returned Node_Temp
1     12:00  Humidity   60.2 # wait, the columns are Node_Temp, Node_Humidity
# Let's fix the print representation:
```
*Actual output:*
```text
  Timestamp         Metric  Value
0     12:00      Node_Temp   25.4
1     12:00  Node_Humidity   60.2
2     13:00      Node_Temp   26.1
3     13:00  Node_Humidity   58.9
```

---

### Worked Example 3: Parsing Monthly Regional Revenue

```python
import pandas as pd

# Regional monthly revenue
df_regional = pd.DataFrame({
    "Region": ["East", "West"],
    "Rev_Jan": [12000, 15000],
    "Rev_Feb": [13000, 14500]
})

# Convert to long format using wide_to_long
df_revenue = pd.wide_to_long(
    df_regional,
    stubnames="Rev",
    i="Region",
    j="Month",
    sep="_",
    suffix=r"\w+"
).reset_index()

print("--- Tidy Regional Revenue ---")
print(df_revenue)
```

### Output

```text
--- Tidy Regional Revenue ---
  Region Month    Rev
0   East   Jan  12000
1   West   Jan  15000
2   East   Feb  13000
3   West   Feb  14500
```

---

## Practice Questions

1. State the three main principles of Tidy Data.
2. Explain the differences between the `.stack()` and `.unstack()` methods in Pandas.
3. Write a command to melt a DataFrame `df` where column `User` is the ID variable and columns `Day1`, `Day2` are unpivoted.
4. How does `pd.wide_to_long()` differ from `pd.melt()` in its core usage?
5. Write a command to stack a DataFrame, keeping missing value cells (`NaN`) in the output Series.
6. What is a MultiIndex, and how is it created during a `.stack()` operation?
7. Write a script that unstacks a MultiIndexed DataFrame along level 0.
8. Explain the purpose of the `suffix` parameter inside `pd.wide_to_long()`.
9. How can you flatten a MultiIndexed row index into regular columns post-stacking?
10. Describe the visual layout of wide-format data versus long-format data.

---

## Mini Assignments

### Assignment 1: Student Attendance Log Reshaping
- Create a classroom register in wide format with columns: `StudentID`, `Monday_Status`, `Tuesday_Status`, and `Wednesday_Status`.
- Melt the DataFrame into a tidy long format where columns are `StudentID`, `Day`, and `Status`.

### Assignment 2: Weather Sensor Metrics Stacking
- Create a weather telemetry DataFrame with columns: `Temp_Max`, `Temp_Min`, `Wind_Max`, and `Wind_Min` indexed by date.
- Stack the metrics and unstack them along the category prefix (e.g. Temp vs Wind) using MultiIndex levels.

### Assignment 3: Regional Monthly Sales Parsing
- Create a store Sales DataFrame with columns: `StoreID`, `Sales_Jan`, `Sales_Feb`, `Profit_Jan`, and `Profit_Feb`.
- Use `pd.wide_to_long()` to restructure the DataFrame into a tidy format with columns `StoreID`, `Month`, `Sales`, and `Profit`.

---

## Interview-Oriented Questions

- **Explain the three principles of Tidy Data and why they are critical for data pipelines.**
  - *Answer*: 1. Each variable forms a column. 2. Each observation forms a row. 3. Each type of observational unit forms a table. Tidy data is critical because it standardizes the data layout, allowing toolsets (like plotting and machine learning packages) to parse variables consistently.
- **What is the difference between stacking (`.stack()`) and melting (`pd.melt()`)?**
  - *Answer*: `.stack()` moves column labels into row indices, converting a DataFrame into a MultiIndexed Series. `pd.melt()` unpivots columns into rows of a flat DataFrame, creating a column for labels and a column for values, which preserves a 2D tabular layout without MultiIndexes.
- **How does `.unstack()` handle duplicate index combinations?**
  - *Answer*: `.unstack()` requires unique index labels for the level being unstacked. If there are duplicates, Pandas cannot resolve the cell locations and raises a `ValueError: Index contains duplicate entries, cannot reshape`.
- **Explain the purpose and usage of `pd.wide_to_long()`.**
  - *Answer*: `pd.wide_to_long()` is designed to parse columns with naming patterns like `stubname_suffix` (e.g. `Sales_2025`, `Sales_2026`). It splits the column name by a separator, unpivots the values, and creates a clean index or column for the suffix.
- **Can we unstack a Series that does not have a MultiIndex?**
  - *Answer*: Yes, unstacking a single-index Series pivots its index labels into columns, returning a 1D row DataFrame.

---

## Teaching Notes for This Chapter

- **Deconstruct Reshaping Visually**: Draw a wide table and trace how column names become row values in a long table.
- **Illustrate MultiIndex levels**: Show how `.stack()` and `.unstack()` change levels (from levels 0, 1, etc.) to help students manage nested indexes.
- **Highlight column naming**: Emphasize that consistent column naming (like using underscores as separators) is crucial for automated reshaping.

---

## Chapter Wrap-up Concepts Students Must Master

- Tidy Data requires one variable per column and one observation per row.
- Use `.stack()` to shift column headers into row indices (making the table taller) and `.unstack()` to shift row indices into column headers (making the table wider).
- Use `pd.melt()` to convert wide-format columns into a flat, long-format table.
- Use `pd.wide_to_long()` to parse columns with systematic naming patterns (like `Sales_2025`).
- Run `.reset_index()` after reshaping to flatten MultiIndexes into regular columns.
