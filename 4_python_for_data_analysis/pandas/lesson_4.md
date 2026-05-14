# Pandas DataFrame Fundamentals

## Lesson Overview

- This chapter introduces the **DataFrame** object, which is the central, most critical data structure in the entire Pandas ecosystem.
- A DataFrame is a two-dimensional, size-mutable, tabular data structure with labeled rows and columns.
- You can conceptualize a DataFrame as a highly optimized spreadsheet, a SQL table, or a dictionary of Pandas Series objects that all share the same index.
- Mastering DataFrames is essential for importing, cleaning, exploring, manipulating, and preparing multi-variable datasets for machine learning and business intelligence workflows.

## Learning Objectives

- Understand the architectural design of a Pandas DataFrame.
- Differentiate between Series and DataFrames, recognizing how Series combine to form a DataFrame.
- Initialize and construct DataFrames from various Python and NumPy data structures.
- Select, slice, and extract subsets of data across rows and columns using `.loc`, `.iloc`, and direct bracket notation.
- Add, rename, modify, and drop columns and rows safely.
- Perform high-performance multi-condition boolean filtering and leverage the chainable `.query()` method.
- Inspect datasets using summary methods like `.info()`, `.describe()`, `.head()`, and `.tail()`.
- Avoid the infamous `SettingWithCopyWarning` by adhering to modern access patterns.
- Apply real-world performance best practices, including PyArrow backends and vectorized execution.

---

## What is a Pandas DataFrame?

- A **Pandas DataFrame** is a two-dimensional labeled data structure.
- It aligns data in a grid of rows and columns, similar to a relational database table or an Excel worksheet.
- Unlike a standard 2D NumPy array, a DataFrame can accommodate heterogeneous data types across its columns (e.g., integers in one column, floats in another, strings/objects in a third, and datetimes in a fourth).

### Core Conceptual Model

- **Row Index**: Every row has a label (the Index). By default, it is a sequential integer sequence starting from 0, but it can be set to custom labels, timestamps, or unique identifiers.
- **Column Index**: Every column has a label (the Column Name).
- **Data Values**: The actual cell values stored in optimized underlying arrays.

### Example

```python
import pandas as pd

data = {
    "Name": ["Amit", "Riya", "Karan", "Neha"],
    "Age": [24, 22, 25, 23],
    "Score": [88.5, 92.0, 79.5, 95.0]
}

df = pd.DataFrame(data)
print(df)
```

### Output

```text
    Name  Age  Score
0   Amit   24   88.5
1   Riya   22   92.0
2  Karan   25   79.5
3   Neha   23   95.0
```

- The leftmost column (`0, 1, 2, 3`) represents the automatically generated row **Index**.
- The top header (`Name, Age, Score`) represents the **Columns**.
- Each individual column extracted from this DataFrame behaves exactly as a Pandas **Series**.

---

## Real-Life Examples of DataFrames

DataFrames represent nearly all tabular data found in industry:
- **E-commerce Transactions**: Columns for Order ID, Customer ID, Timestamp, Product Name, Quantity, and Total Price.
- **Financial Stock Market Data**: Columns for Date, Open, High, Low, Close, and Volume.
- **Human Resources Records**: Columns for Employee ID, Name, Department, Joining Date, and Salary.
- **Healthcare Patient Log**: Columns for Patient ID, Age, Blood Group, Diagnosis, and Admission Status.
- **Sensor Telemetry**: Columns for Timestamp, Sensor ID, Temperature, Humidity, and Battery Level.

---

## Series vs DataFrame Anatomy

| Feature | Pandas Series | Pandas DataFrame |
| :--- | :--- | :--- |
| **Dimensionality** | 1-Dimensional | 2-Dimensional |
| **Structure** | Single column of labeled data | Table of multiple labeled columns |
| **Data Types** | Homogeneous (single dtype per Series) | Heterogeneous across columns (each column has its own dtype) |
| **Primary Access** | Access by row label/position | Access by column name, then row label/position |
| **Analogy** | A single column in an Excel sheet | The entire Excel worksheet |

---

## Anatomy of a DataFrame

A DataFrame object is composed of several foundational structural attributes:

- **`.index`**: The row labels.
- **`.columns`**: The column labels.
- **`.values`** / **`.to_numpy()`**: The underlying 2D array of data values.
- **`.dtypes`**: The specific data type of each individual column.
- **`.shape`**: A tuple representing `(number_of_rows, number_of_columns)`.
- **`.size`**: The total number of data elements (`rows * columns`).

### Structural Inspection Example

```python
import pandas as pd

df = pd.DataFrame({
    "Product": ["Laptop", "Tablet", "Smartphone"],
    "Price": [75000, 32000, 45000],
    "Stock": [15, 30, 50]
}, index=["P01", "P02", "P03"])

print("--- DataFrame ---")
print(df)

print("\nIndex:", df.index)
print("Columns:", df.columns)
print("Shape:", df.shape)
print("Size:", df.size)
print("\nData Types:\n", df.dtypes)
```

### Output

```text
--- DataFrame ---
        Product  Price  Stock
P01      Laptop  75000     15
P02      Tablet  32000     30
P03  Smartphone  45000     50

Index: Index(['P01', 'P02', 'P03'], dtype='object')
Columns: Index(['Product', 'Price', 'Stock'], dtype='object')
Shape: (3, 3)
Size: 9

Data Types:
 Product    object
Price       int64
Stock       int64
dtype: object
```

---

## Creating a DataFrame

Pandas provides extensive flexibility for initializing DataFrames from multiple native Python and numerical structures.

### 1. From a Dictionary of Lists (Most Common)

Keys become column headers, and the associated lists become column data. All lists must be of identical length.

```python
import pandas as pd

sales_data = {
    "Region": ["North", "South", "East", "West"],
    "Target": [50000, 60000, 45000, 55000],
    "Achieved": [52000, 58000, 48000, 53000]
}
df_sales = pd.DataFrame(sales_data)
print(df_sales)
```

### 2. From a List of Dictionaries

Each dictionary in the list represents a single row. Keys map to column names. Missing keys in specific rows are automatically populated with `NaN`.

```python
users = [
    {"User": "admin", "Role": "Administrator", "Active": True},
    {"User": "guest", "Role": "Viewer"},  # Missing 'Active' key
    {"User": "editor", "Role": "Contributor", "Active": False}
]
df_users = pd.DataFrame(users)
print(df_users)
```

### Output

```text
     User           Role  Active
0   admin  Administrator    True
1   guest         Viewer     NaN
2  editor    Contributor   False
```

### 3. From a 2D NumPy Array

Pass the array along with optional column names and row index labels.

```python
import numpy as np
import pandas as pd

matrix = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
df_matrix = pd.DataFrame(matrix, columns=["Col_A", "Col_B", "Col_C"], index=["R1", "R2", "R3"])
print(df_matrix)
```

### 4. From External Files (Modern High-Performance I/O)

In modern Pandas workflows, loading external data utilizing optimized parsing backends is standard practice. For example, using the **PyArrow** engine drastically accelerates parsing speed for large CSV files.

```python
# Loading CSV using standard PyArrow acceleration
# df = pd.read_csv("large_dataset.csv", engine="pyarrow")
```

### 5. Creating an Empty DataFrame

```python
df_empty = pd.DataFrame()
print(df_empty.empty)  # Returns True
```

---

## Accessing Columns and Rows

Accessing data precisely and efficiently is one of the most critical skills when working with DataFrames.

### Accessing Columns

#### Selecting a Single Column

Selecting a single column returns a **Pandas Series**.

```python
df = pd.DataFrame({
    "Item": ["Mouse", "Keyboard", "Monitor"],
    "Price": [500, 1200, 8500]
})

# Preferred bracket notation
col_series = df["Price"]
print(type(col_series))  # <class 'pandas.core.series.Series'>
print(col_series)
```

> **Note on Dot Notation**: You can access columns as attributes (e.g., `df.Price`), but this practice is **strongly discouraged** in production code. Dot notation fails if the column name contains spaces, matches a built-in DataFrame method name (like `count`, `mean`, or `index`), or starts with a number. Always prefer explicit bracket notation `df["Column_Name"]`.

#### Selecting Multiple Columns

Pass a list of column names inside the indexing brackets. This returns a subset **DataFrame**.

```python
subset_df = df[["Item", "Price"]]
print(type(subset_df))  # <class 'pandas.core.frame.DataFrame'>
print(subset_df)
```

---

### Accessing Rows using `.loc` and `.iloc`

Pandas enforces a strict separation between **label-based** selection and **integer-position-based** selection to prevent ambiguity.

- **`.loc[]`**: Selects rows and columns by their **explicit label** (index names and column names).
- **`.iloc[]`**: Selects rows and columns by their **zero-indexed integer position** (exactly like Python list slicing).

#### Setup for Demonstration

```python
import pandas as pd

df_emp = pd.DataFrame({
    "Age": [29, 34, 41, 25],
    "Dept": ["IT", "HR", "Engineering", "Marketing"],
    "Salary": [85000, 72000, 110000, 58000]
}, index=["E101", "E102", "E103", "E104"])

print(df_emp)
```

#### Using `.loc` (Label-Based)

```python
# Extract single row as a Series
print("--- Single Row loc ---")
print(df_emp.loc["E102"])

# Extract specific rows and specific columns simultaneously
print("\n--- Multi Row & Column loc ---")
print(df_emp.loc[["E101", "E103"], ["Dept", "Salary"]])

# Slicing with labels (Note: Ending label is INCLUSIVE in .loc)
print("\n--- Slicing loc ---")
print(df_emp.loc["E101":"E103", "Salary"])
```

#### Using `.iloc` (Position-Based)

```python
# Extract row at integer index 1 (second row)
print("--- Single Row iloc ---")
print(df_emp.iloc[1])

# Extract specific rows and columns by index integers
print("\n--- Multi Row & Column iloc ---")
print(df_emp.iloc[[0, 2], [1, 2]])

# Slicing by integer position (Note: Ending position is EXCLUSIVE in .iloc)
print("\n--- Slicing iloc ---")
print(df_emp.iloc[0:3, 0:2])
```

#### Extracting Scalar Values precisely

For accessing a single precise cell value, use `.loc[row, col]` or `.iloc[row_idx, col_idx]`. 

```python
# Direct scalar access
tech_dept = df_emp.loc["E101", "Dept"]
print("Department of E101:", tech_dept)
```

---

## Adding, Modifying, and Deleting Columns/Rows

DataFrames are fully mutable structures. You can append new features, transform existing ones, or purge obsolete slices.

### Adding a New Column

#### Method 1: Direct Assignment (In-Place)

Assigning a Series, list, or scalar to a new column key creates the column automatically.

```python
df_emp["Bonus"] = [8500, 7200, 11000, 5800]
df_emp["Country"] = "India"  # Broadcasts scalar value to all rows
print(df_emp)
```

#### Method 2: Using `.assign()` (Functional, Non-Destructive)

`.assign()` returns a **new copy** of the DataFrame with the updated or added columns, leaving the original intact. This is excellent for clean method chaining workflows.

```python
df_new = df_emp.assign(
    Total_Comp=lambda x: x["Salary"] + x["Bonus"],
    Tax_Bracket=lambda x: x["Salary"] > 80000
)
print(df_new[["Salary", "Bonus", "Total_Comp", "Tax_Bracket"]])
```

#### Method 3: Using `.insert()` (Specific Position)

Inserts a column in-place at a specified integer index position.

```python
# Insert 'Status' column at integer index 1
df_emp.insert(1, "Status", ["Full-Time", "Full-Time", "Contract", "Intern"])
print(df_emp)
```

---

### Modifying Column Values Safely

Always modify values using `.loc` to target slices cleanly. Avoid chained index modifications.

```python
# Give a raise to employees in the IT department safely
df_emp.loc[df_emp["Dept"] == "IT", "Salary"] = 95000
print(df_emp.loc["E101"])
```

---

### Deleting Columns and Rows

Use the `.drop()` method. By default, `.drop()` returns a new DataFrame copy. Use `inplace=True` or reassign the variable to persist changes.

- **Drop Columns**: Set axis parameter to `1` or use the explicit `columns` keyword argument.
- **Drop Rows**: Set axis parameter to `0` or use the explicit `index` keyword argument.

```python
# Dropping columns
df_dropped_cols = df_emp.drop(columns=["Country", "Status"])
print("--- Columns Dropped ---")
print(df_dropped_cols.columns)

# Dropping rows by index labels
df_dropped_rows = df_emp.drop(index=["E102", "E104"])
print("\n--- Rows Dropped ---")
print(df_dropped_rows.index)
```

Alternatively, you can use the standard Python `del` statement to permanently delete a column in-place:
```python
del df_emp["Country"]
```

---

## Inspection and Summary Methods

When working with production datasets containing millions of rows, direct printing is impractical. Pandas provides targeted diagnostic methods.

### Quick Peek Methods

- **`.head(n)`**: Returns the first `n` rows (default is 5).
- **`.tail(n)`**: Returns the last `n` rows (default is 5).
- **`.sample(n)`**: Returns a random sample of `n` rows. Useful for validating randomized data sets.

```python
import pandas as pd
df_large = pd.DataFrame({"Val": range(1000)})
print("Head 2:\n", df_large.head(2))
print("Tail 2:\n", df_large.tail(2))
print("Sample 2:\n", df_large.sample(2, random_state=42))
```

### Deep Inspection Methods

#### `.info()`

Provides a comprehensive technical summary of the DataFrame: row index range, column names, non-null counts, column data types, and total memory usage.

```python
df_emp.info()
```

### Output Example

```text
<class 'pandas.core.frame.DataFrame'>
Index: 4 entries, E101 to E104
Data columns (total 5 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   Age     4 non-null      int64 
 1   Status  4 non-null      object
 2   Dept    4 non-null      object
 3   Salary  4 non-null      int64 
 4   Bonus   4 non-null      int64 
dtypes: int64(3), object(2)
memory usage: 360.0+ bytes
```

#### `.describe()`

Generates summary descriptive statistics for numeric columns (count, mean, standard deviation, min, 25th percentile, median, 75th percentile, max). Passing `include='all'` or `include='object'` provides frequency summaries for categorical/text columns.

```python
print(df_emp.describe())
```

---

## Filtering and Boolean Indexing

Boolean indexing allows slicing rows based on dynamic logical conditions applied to column data.

### Single Condition Filtering

```python
import pandas as pd

df = pd.DataFrame({
    "Product": ["A", "B", "C", "D", "E"],
    "Sales": [120, 85, 300, 45, 190],
    "Rating": [4.2, 3.8, 4.9, 2.5, 4.6]
})

# Create a boolean mask where Sales exceed 100
mask = df["Sales"] > 100
print("Boolean Mask:\n", mask)

# Apply mask to extract rows
high_sales = df[mask]
print("\nFiltered DataFrame:\n", high_sales)
```

### Multiple Condition Filtering

Combine conditions using bitwise operators. **Every individual condition must be enclosed in parentheses**.
- **`&`** : Bitwise AND
- **`|`** : Bitwise OR
- **`~`** : Bitwise NOT

```python
# Find highly rated products with substantial sales volume
top_performers = df[(df["Sales"] > 100) & (df["Rating"] >= 4.5)]
print("--- Top Performers ---")
print(top_performers)

# Find underperforming or poorly reviewed items
flagged = df[(df["Sales"] < 50) | (df["Rating"] < 3.0)]
print("\n--- Flagged Items ---")
print(flagged)
```

---

### Modern Query-Based Filtering (`.query()`)

The `.query()` method allows filtering using clean string expressions. It improves code readability, avoids repeated variable typing, and executes efficiently using optimized C backends like `numexpr`.

```python
# Equivalent query expression for multiple conditions
queried_df = df.query("Sales > 100 and Rating >= 4.5")
print("--- Queried Results ---")
print(queried_df)

# Referencing external local variables in queries using the '@' prefix
min_sales_threshold = 150
dynamic_query = df.query("Sales >= @min_sales_threshold")
print("\n--- Dynamic Variable Query ---")
print(dynamic_query)
```

---

## Index Alignment and Broadcasting

When performing arithmetic operations between two DataFrames or between a DataFrame and a Series, Pandas automatically aligns structures by their row and column index labels.

### Broadcasting a Scalar or Array

Applying a scalar operation broadcasts to every cell.

```python
df_nums = pd.DataFrame({"A": [10, 20], "B": [30, 40]})
print("Original:\n", df_nums)
print("\nMultiplied by 10:\n", df_nums * 10)
```

### DataFrame-to-DataFrame Alignment

If labels do not match perfectly between structures, missing elements in the aligned result are populated with `NaN`.

```python
df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=[0, 1])
df2 = pd.DataFrame({"B": [5, 6], "C": [7, 8]}, index=[1, 2])

# Aligns indices 0, 1, 2 and columns A, B, C automatically
res = df1 + df2
print("Aligned Addition:\n", res)
```

### Output

```text
    A     B   C
0 NaN   NaN NaN
1 NaN   9.0 NaN
2 NaN   NaN NaN
```
- Row `0` exists only in `df1`.
- Row `2` exists only in `df2`.
- Column `A` exists only in `df1`, Column `C` only in `df2`.
- Perfect overlap occurs only at row `1`, column `B` (`4 + 5 = 9.0`).

To avoid losing non-overlapping data, use explicit functional math methods with `fill_value`:
```python
clean_res = df1.add(df2, fill_value=0)
print("Safer Addition:\n", clean_res)
```

---

## Handling Missing Data Basics

Real-world tables are riddled with incomplete records. DataFrames provide streamlined API workflows for missing value treatment.

- **`.isna()`** / **`.isnull()`**: Returns a matching boolean grid identifying null cells.
- **`.dropna()`**: Strips out rows or columns containing missing values.
- **`.fillna(val)`**: Replaces missing entries with targeted default values or calculated metrics.

```python
import numpy as np
import pandas as pd

df_miss = pd.DataFrame({
    "A": [1.0, np.nan, 3.0, 4.0],
    "B": [np.nan, 2.0, 3.0, np.nan],
    "C": ["Doc", "Log", None, "PDF"]
})

print("--- Original Missing Set ---")
print(df_miss)

print("\n--- Missing Cell Counts per Column ---")
print(df_miss.isna().sum())

print("\n--- Dropping Rows with Any Nulls ---")
print(df_miss.dropna())

print("\n--- Filling Nulls with Defaults ---")
# Fill numeric columns with column averages, text with placeholder strings
filled = df_miss.assign(
    A=lambda x: x["A"].fillna(x["A"].mean()),
    B=lambda x: x["B"].fillna(0),
    C=lambda x: x["C"].fillna("Unknown")
)
print(filled)
```

---

## Modern Best Practices & Performance Optimization

Adhering to high-performance Pandas development standards ensures maintainable codebases that execute efficiently on large datasets.

### 1. Avoid Chained Indexing (`SettingWithCopyWarning`)

**Chained Indexing** occurs when you execute two consecutive indexing operations (e.g., `df[condition]["Column"] = value`). This triggers the **`SettingWithCopyWarning`** because Pandas cannot guarantee whether the intermediate result is a slice view or a completely disconnected memory copy, risking silent assignment failures.

#### The Wrong Way (Triggers Warning)
```python
# Avoid this pattern completely
# df_emp[df_emp["Dept"] == "HR"]["Salary"] = 75000
```

#### The Right Way (Guaranteed and Efficient)
Target the slice simultaneously using `.loc[row_indexer, column_indexer]`.
```python
df_emp.loc[df_emp["Dept"] == "HR", "Salary"] = 75000
```

### 2. Leverage Vectorization Over Iteration

Never use standard Python `for` loops or `.iterrows()` to transform or compute columns unless absolutely forced by external API constraints. Vectorized operations execute via underlying highly optimized C arrays, achieving speeds thousands of times faster than iterative logic.

```python
# Bad: Slow, iterative loop
# for idx, row in df.iterrows():
#     df.loc[idx, "Total"] = row["Price"] * row["Qty"]

# Good: Direct vectorized execution
# df["Total"] = df["Price"] * df["Qty"]
```

### 3. Downcast Numeric Types Early

Default integer (`int64`) and floating-point (`float64`) allocations consume generous memory footprints. Downcasting numeric columns directly scales memory efficiency.

```python
df_opt = pd.DataFrame({"Small_Ints": [1, 2, 3, 4, 5]})
print("Original Memory:", df_opt["Small_Ints"].memory_usage())

# Downcast to 8-bit integer safely
df_opt["Small_Ints"] = pd.to_numeric(df_opt["Small_Ints"], downcast="integer")
print("Optimized Memory:", df_opt["Small_Ints"].memory_usage())
```

---

## Common Mistakes Students Make

- **Using Dot Notation for Column Creation**: Executing `df.New_Col = [1, 2, 3]` assigns a general dynamic attribute to the Python object shell, **not** a real accessible column inside the tabular data structure. Always use brackets: `df["New_Col"] = [1, 2, 3]`.
- **Forgetting Parentheses in Complex Filters**: Writing `df[df["A"] > 10 & df["B"] < 5]` triggers logical runtime exceptions due to Python's evaluation order precedence. Bitwise operators bind tighter than comparison symbols. Use explicit grouping: `df[(df["A"] > 10) & (df["B"] < 5)]`.
- **Assuming Changes are In-Place by Default**: Calling methods like `df.drop(columns=["ColA"])` or `df.fillna(0)` leaves the source DataFrame unedited unless explicitly caught with a re-assignment (`df = df.drop(...)`) or flagged via `inplace=True`.
- **Confusing `.loc` and `.iloc` Slicing Boundaries**: Forgetting that `df.loc["R1":"R3"]` includes row `"R3"`, whereas `df.iloc[0:3]` stops immediately prior to integer index `3`.
- **Mixing Up Axis Arguments**: Confusing `axis=0` (operations traveling across/collapsing row indices vertically) with `axis=1` (operations traversing column axes horizontally).

---

## Best Practices

- Standardize column names early using list comprehensions or vectorized string functions to remove whitespace and maintain clean casing (e.g., `df.columns = df.columns.str.strip().str.lower()`).
- Always run `.info()` immediately after parsing new data files to confirm correct type interpretations and check for unexpected missing values.
- Practice clean functional formatting via `.assign()` method chaining to ensure analytical workflows remain transparent and reproducible.
- Utilize `.query()` when composing dynamic logic strings containing multiple overlapping parameters.
- Enforce schema validation on critical data pipelines using frameworks like **Pandera** to guarantee clean column typing before running internal models.

---

## Worked Real-World Examples

### Worked Example 1: Student Database Processing

```python
import pandas as pd

# 1. Initialize data
students = {
    "StudentID": ["S101", "S102", "S103", "S104", "S105"],
    "Name": ["Aarav", "Priya", "Vikram", "Ananya", "Rohan"],
    "Math": [88, 92, 45, 78, 62],
    "Science": [91, 89, 52, 84, 55],
    "English": [85, 95, 60, 88, 70]
}

df_students = pd.DataFrame(students).set_index("StudentID")

# 2. Add structural calculated metrics via assignment
df_processed = df_students.assign(
    Total_Score=lambda x: x["Math"] + x["Science"] + x["English"],
    Average=lambda x: x["Total_Score"] / 3.0,
    Status=lambda x: ["Pass" if avg >= 60 else "Review Required" for avg in x["Average"]]
)

print("--- Processed Student Dashboard ---")
print(df_processed[["Name", "Total_Score", "Average", "Status"]])

# 3. Filter top-tier performers cleanly
top_students = df_processed.query("Average >= 85")
print("\n--- Top Tier Excellence List ---")
print(top_students[["Name", "Average"]])
```

### Output

```text
--- Processed Student Dashboard ---
             Name  Total_Score    Average           Status
StudentID                                                 
S101        Aarav          264  88.000000             Pass
S102        Priya          276  92.000000             Pass
S103       Vikram          157  52.333333  Review Required
S104       Ananya          250  83.333333             Pass
S105        Rohan          187  62.333333             Pass

--- Top Tier Excellence List ---
            Name  Average
StudentID                
S101       Aarav     88.0
S102       Priya     92.0
```

---

### Worked Example 2: Retail Inventory Audit

```python
import pandas as pd

inventory = [
    {"SKU": "K101", "Item": "Mechanical Keyboard", "Cost": 2500, "Qty": 45},
    {"SKU": "M202", "Item": "Wireless Mouse", "Cost": 1200, "Qty": 12},  # Low stock
    {"SKU": "H303", "Item": "Gaming Headset", "Cost": 4500, "Qty": 5},   # Critical stock
    {"SKU": "P404", "Item": "Mousepad", "Cost": 400, "Qty": 150}
]

df_inv = pd.DataFrame(inventory)

# Calculate total inventory capital value vectorized
df_inv["Capital_Value"] = df_inv["Cost"] * df_inv["Qty"]

# Filter out low stock items requiring urgent purchase orders
reorder_target = df_inv[df_inv["Qty"] < 20]

print("--- Total Warehouse Catalog ---")
print(df_inv)

print("\n--- Urgent Reorder Alerts (Qty < 20) ---")
print(reorder_target[["SKU", "Item", "Qty"]])
```

### Output

```text
--- Total Warehouse Catalog ---
    SKU                 Item  Cost  Qty  Capital_Value
0  K101  Mechanical Keyboard  2500   45         112500
1  M202       Wireless Mouse  1200   12          14400
2  H303       Gaming Headset  4500    5          22500
3  P404             Mousepad   400  150          60000

--- Urgent Reorder Alerts (Qty < 20) ---
    SKU            Item  Qty
1  M202  Wireless Mouse   12
2  H303  Gaming Headset    5
```

---

### Worked Example 3: Missing Sales Record Recovery

```python
import numpy as np
import pandas as pd

raw_sales = pd.DataFrame({
    "Store_ID": ["ST_01", "ST_02", "ST_03", "ST_04", "ST_05"],
    "City": ["Mumbai", "Delhi", None, "Bengaluru", "Chennai"],
    "Q1_Rev": [12.5, np.nan, 14.2, 18.0, 9.8],
    "Q2_Rev": [13.1, 15.0, np.nan, 19.5, 10.2]
})

print("--- Raw Audit Log with Incomplete Fields ---")
print(raw_sales)

# Imputation Workflow:
# 1. Fill missing regional codes with unallocated labels
# 2. Impute quarterly gaps with median values observed across functional branches
cleaned_sales = raw_sales.assign(
    City=lambda x: x["City"].fillna("Unallocated"),
    Q1_Rev=lambda x: x["Q1_Rev"].fillna(x["Q1_Rev"].median()),
    Q2_Rev=lambda x: x["Q2_Rev"].fillna(x["Q2_Rev"].median())
)

# Compute annual run-rate estimates
cleaned_sales["Est_H1_Total"] = cleaned_sales["Q1_Rev"] + cleaned_sales["Q2_Rev"]

print("\n--- Fully Recovered Analytics Matrix ---")
print(cleaned_sales)
```

### Output

```text
--- Raw Audit Log with Incomplete Fields ---
  Store_ID       City  Q1_Rev  Q2_Rev
0    ST_01     Mumbai    12.5    13.1
1    ST_02      Delhi     NaN    15.0
2    ST_03       None    14.2     NaN
3    ST_04  Bengaluru    18.0    19.5
4    ST_05    Chennai     9.8    10.2

--- Fully Recovered Analytics Matrix ---
  Store_ID         City  Q1_Rev  Q2_Rev  Est_H1_Total
0    ST_01       Mumbai   12.50   13.10         25.60
1    ST_02        Delhi   13.35   15.00         28.35
2    ST_03  Unallocated   14.20   14.05         28.25
3    ST_04    Bengaluru   18.00   19.50         37.50
4    ST_05      Chennai    9.80   10.20         20.00
```

---

### Worked Example 4: Conditional Promotion Execution

```python
import pandas as pd

df_staff = pd.DataFrame({
    "Emp_ID": [101, 102, 103, 104],
    "Name": ["Rajesh", "Suman", "Kavita", "Tariq"],
    "Rating": [4.8, 3.2, 4.6, 2.9],
    "Base_Salary": [60000, 55000, 70000, 48000]
})

# Give a flat 15% salary bump to staff with appraisal ratings strictly over 4.5
df_staff.loc[df_staff["Rating"] > 4.5, "Base_Salary"] = df_staff["Base_Salary"] * 1.15

print("--- Updated Compensation Register ---")
print(df_staff)
```

### Output

```text
--- Updated Compensation Register ---
   Emp_ID    Name  Rating  Base_Salary
0     101  Rajesh     4.8      69000.0
1     102   Suman     3.2      55000.0
2     103  Kavita     4.6      80500.0
3     104   Tariq     2.9      48000.0
```

---

### Worked Example 5: Combining Multi-Type Access Patterns

```python
import pandas as pd

df_mix = pd.DataFrame({
    "Model": ["Sedan", "SUV", "Hatchback", "Coupe"],
    "Units_Sold": [1200, 2500, 3100, 450],
    "Revenue_Cr": [144.0, 500.0, 186.0, 90.0]
}, index=["M1", "M2", "M3", "M4"])

# 1. Extract specific cell via explicit string coordinate labeling
suv_rev = df_mix.loc["M2", "Revenue_Cr"]

# 2. Extract last two rows using pure integer slicing mechanics
bottom_slice = df_mix.iloc[-2:]

print("SUV Revenue direct hit:", suv_rev)
print("\nBottom Slice inspection via iloc:\n", bottom_slice)
```

### Output

```text
SUV Revenue direct hit: 500.0

Bottom Slice inspection via iloc:
         Model  Units_Sold  Revenue_Cr
M3  Hatchback        3100       186.0
M4      Coupe         450        90.0
```

---

## Practice Questions

1. Construct a Pandas DataFrame populated from a native Python dictionary containing three distinct keys representing arbitrary product classifications.
2. Formulate the syntax necessary to safely extract the second column of a DataFrame as an independent 1D Series object.
3. Write an expression utilizing `.iloc[]` mechanics to isolate the internal structural subset comprising rows `1` through `3` paired exclusively with columns `0` and `2`.
4. Compose a filtering workflow applying bitwise parameters capable of extracting entries whose baseline prices drop below `500` simultaneously displaying stock flags marked true.
5. Create an analytical command updating missing text cell contents to read `"Review Pending"` using robust vectorized methods.
6. Contrast the analytical capabilities exposed by executing `.info()` versus those supplied by calling `.describe()`.
7. Formulate a logical string input structured for integration inside `.query()` to isolate records tracking sales metrics surpassing defined baseline targets.
8. Outline the procedural pipeline required to drop an internal feature axis permanently from memory space.
9. Explain the technical root source triggering the `SettingWithCopyWarning` during analytical scripting workflows.
10. Demonstrate how to insert a descriptive string tag key directly into index position `2` across an active dataset grid.

---

## Mini Assignments

### Assignment 1: Corporate Expense Analytics
- Scaffold an internal expense record table containing fields tracing `Department`, `Monthly_Budget`, and `Actual_Spend` across 8 active corporate sectors.
- Generate a dynamically calculated column identifying net financial budget variances (`Monthly_Budget - Actual_Spend`).
- Apply boolean masking workflows to separate out operational branches flagged for overspending.
- Calculate summary descriptive metrics isolating standard average expenditures incurred across standard operations.

### Assignment 2: Real Estate Portfolio Management
- Load a property ledger collection logging structural properties tracking `Property_ID`, `City`, `Square_Feet`, and `Listing_Price`. Include deliberate `NaN` missing fields across target spatial inputs.
- Cleanse structural gaps by dropping unrecoverable listings missing valid unique identifier keys.
- Impute missing listing valuation estimates substituting baseline metric pricing medians computed across local structural zones.
- Filter the target portfolio matrix extracting large luxury footprints scaling above 2500 square feet carrying listing rates beneath market norms.

### Assignment 3: Fleet Maintenance Logging
- Instantiate a vehicle status frame containing vehicle codes, travel distances, and mechanical service statuses.
- Execute direct scalar assignment statements updating service flags to show true for specific vehicle target rows targeted safely via label-based indexing.
- Generate summary categorical counts highlighting functional frequency counts tracking distribution metrics across operational fleet classifications.
- Sort the ultimate analytics dataset vertically prioritizing structural entries boasting highest travel logs arranged sequentially downward.

---

## Interview-Oriented Questions

- **How do you conceptualize the structural link tying Pandas Series objects to core DataFrames?**
  - *Answer*: A DataFrame is structurally engineered as a managed dictionary collection containing underlying Series vectors that are bound together horizontally sharing a unified mapping index.
- **What programmatic mechanics segregate the functionality provided by `.loc` access patterns from `.iloc` execution paths?**
  - *Answer*: `.loc` interfaces operate entirely via semantic string/label coordination maps (inclusive slicing). `.iloc` interfaces rely strictly upon sequential integer addressing arrays acting directly on internal buffer spaces (exclusive slicing).
- **Why should developers avoid utilizing dot notation access styles when configuring programmatic feature assignments?**
  - *Answer*: Dot style assignment patterns risk silent failures if variable strings overlap reserved functional API keywords or contain structural whitespace separators, resulting in broken frame dependencies.
- **Explain the exact trigger causing Pandas to raise a `SettingWithCopyWarning` during data editing tasks.**
  - *Answer*: The warning fires whenever analytical workflows process chained square bracket sub-selections (`df[mask]["col"] = val`), creating uncertain downstream reference paths unable to verify whether source memory blocks receive requested updates safely.
- **What optimizations justify using the PyArrow read engine when ingesting large production files over standard parsing defaults?**
  - *Answer*: Integrating Arrow backends accelerates parsing via zero-copy CPU structures, advanced multi-threaded block reading routines, and robust memory allocations minimizing dynamic RAM overhead.

---

## Teaching Notes for This Chapter

- **Scaffold Progression**: Begin lectures utilizing simple dictionary construction models prior to introducing nested array operations to solidify visual structure mechanics.
- **Emphasize Safe Indexing**: Devote substantial review cycles verifying student comprehension regarding `.loc` assignments. Demonstrate code examples generating warnings live to clear up common misconceptions early.
- **Encourage Method Chaining**: Frame data manipulation as a flowing programmatic transformation pipeline. Prompt students to practice writing multi-step analytical updates inside enclosed `.assign()` wrappers.
- **Contextualize Real-World Chaos**: Reinforce the necessity of calling `.info()` and validating non-null distribution sets before writing modeling code, highlighting that raw data is rarely pristine.

---

## Chapter Wrap-up Concepts Students Must Master

- DataFrames represent 2D tabular data arrays containing indexed rows intersecting named columns.
- The underlying columns operate natively as distinct single-dimensional Series structures sharing common index labels.
- Slicing and targeted selections require explicit execution using label-driven `.loc[]` blocks or integer-based `.iloc[]` bounds.
- Multi-conditional boolean masking requires explicit bitwise logical operators (`&`, `|`) paired with enclosed logical expressions.
- Functional filtering strategies utilizing `.query()` expressions optimize complex querying workflows.
- Programmatic modifications must target data spaces using robust, unified coordinate indexing to guarantee data integrity and bypass warning flags.
- Comprehensive descriptive profiles are immediately available via built-in `.info()` and `.describe()` diagnostic methods.
- Vectorized computations applied directly across feature arrays scale computational efficiency over manual iteration logic.
