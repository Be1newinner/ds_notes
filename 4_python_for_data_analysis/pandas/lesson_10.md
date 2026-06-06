# Working with Missing Data (detecting, dropna, fillna)

## Lesson Overview

- This chapter covers the detection, removal, and imputation of missing values in Pandas. Handling missing data is one of the most critical steps in data preprocessing.
- Datasets collected from the real world are rarely complete. They contain gaps due to missing database entries, failed sensor readings, or incomplete survey responses. If ignored, these missing values (`NaN`) can lead to runtime errors, bias statistical summaries, and degrade machine learning model performance.
- We will cover standard functions like `.isna()`, `.dropna()`, and `.fillna()`, and explore newer nullable data types (like `Int64`) introduced to handle integers with missing records without converting them to float values.
- Mastering missing data workflows ensures that downstream analytical steps operate on clean, mathematically sound tables.

## Learning Objectives

- Detect and count missing values at the Series and DataFrame level using `.isna()` and `.notna()`.
- Drop incomplete records selectively using `.dropna()` based on axis, threshold counts, or target subsets.
- Impute missing entries with `.fillna()` using constants, column-specific mappings, or time-series fill methods (`ffill`, `bfill`).
- Replace specific dummy placeholders (e.g., `-999`, `"N/A"`) with standard NumPy `NaN` values.
- Understand the implications of standard floating-point `NaN` representation versus Pandas nullable integer/boolean types.

---

## Representing Missing Data in Pandas

Traditionally, Pandas represented missing data using floating-point `NaN` (Not a Number) values from NumPy.
- **Side Effect**: Because `NaN` is a float, loading an integer column that contains missing values forces the entire column to be cast to `float64`.
- **Modern Alternative**: Pandas introduces nullable integer types (like `Int64` with a capital `I`) and nullable booleans (`boolean`), which allow missing values to exist without type casting.

### Setup for Demonstration

```python
import pandas as pd
import numpy as np

# Create dataset with missing values represented in multiple ways
df_missing = pd.DataFrame({
    "Product": ["Laptop", "Mouse", "Keyboard", "Monitor", "Chair"],
    "Stock": [15, np.nan, 42, 8, np.nan],
    "Price": [1200.00, 25.50, np.nan, 350.00, np.nan],
    "Promo_Code": ["SAVE10", "N/A", "WELCOME", np.nan, "FREE"]
})

print("--- Master Incomplete Table ---")
print(df_missing)
```

### Output

```text
--- Master Incomplete Table ---
    Product  Stock   Price Promo_Code
0    Laptop   15.0  1200.0     SAVE10
1     Mouse    NaN    25.5        N/A
2  Keyboard   42.0     NaN    WELCOME
3   Monitor    8.0   350.0        NaN
4     Chair    NaN     NaN       FREE
```

---

## 1. Detecting Missing Values

Use `.isna()` (or its alias `.isnull()`) and `.notna()` (or `.notnull()`) to identify missing entries.

```python
# Check for nulls element-wise
print("--- Is Null Mask ---")
print(df_missing.isna())

# Count missing values per column
print("\n--- Count of Missing Values per Column ---")
print(df_missing.isna().sum())

# Calculate percentage of missing values per column
print("\n--- Percentage of Missing Values ---")
print(df_missing.isna().mean() * 100)
```

### Output

```text
--- Is Null Mask ---
   Product  Stock  Price  Promo_Code
0    False  False  False       False
1    False   True  False       False
2    False  False   True       False
3    False  False  False        True
4    False   True   True       False

--- Count of Missing Values per Column ---
Product       0
Stock         2
Price         2
Promo_Code    1
dtype: int64

--- Percentage of Missing Values ---
Product        0.0
Stock         40.0
Price         40.0
Promo_Code    20.0
dtype: float64
```

---

## 2. Dropping Missing Data with `.dropna()`

Use `.dropna()` to remove rows or columns containing missing values.

### Parameters
- `axis`: `0` or `'index'` to drop rows; `1` or `'columns'` to drop columns.
- `how`: `'any'` (default, drops row/column if any value is null); `'all'` (drops row/column only if all values are null).
- `subset`: List of columns to inspect for missing values.
- `thresh`: Keep only rows/columns that contain at least `thresh` non-null values.

```python
# Drop rows where ANY value is missing
drop_any = df_missing.dropna()
print("--- Dropped Any Null Rows ---")
print(drop_any)

# Drop rows only if Stock or Price is missing
drop_subset = df_missing.dropna(subset=["Stock", "Price"])
print("\n--- Dropped Rows Based on Subset [Stock, Price] ---")
print(drop_subset)

# Keep rows with at least 3 non-null columns
drop_thresh = df_missing.dropna(thresh=3)
print("\n--- Dropped Rows (Threshold >= 3 Non-Nulls) ---")
print(drop_thresh)
```

### Output

```text
--- Dropped Any Null Rows ---
   Product  Stock   Price Promo_Code
0   Laptop   15.0  1200.0     SAVE10

--- Dropped Rows Based on Subset [Stock, Price] ---
   Product  Stock   Price Promo_Code
0   Laptop   15.0  1200.0     SAVE10
3  Monitor    8.0   350.0        NaN

--- Dropped Rows (Threshold >= 3 Non-Nulls) ---
    Product  Stock   Price Promo_Code
0    Laptop   15.0  1200.0     SAVE10
1     Mouse    NaN    25.5        N/A
2  Keyboard   42.0     NaN    WELCOME
3   Monitor    8.0   350.0        NaN
```

---

## 3. Imputing Missing Values with `.fillna()`

Use `.fillna()` to replace missing values with a designated constant or calculated metric.

```python
# Copy the dataset
df_fill = df_missing.copy()

# Fill all NaNs with a single constant value
print("--- Constant Fill ---")
print(df_fill.fillna(0))

# Fill using a dictionary to specify column-specific values
fill_dict = {
    "Stock": df_fill["Stock"].median(),
    "Price": df_fill["Price"].mean(),
    "Promo_Code": "NONE"
}
print("\n--- Column-Specific Fill ---")
print(df_fill.fillna(value=fill_dict))
```

### Output

```text
--- Constant Fill ---
    Product  Stock   Price Promo_Code
0    Laptop   15.0  1200.0     SAVE10
1     Mouse    0.0    25.5        N/A
2  Keyboard   42.0     0.0    WELCOME
3   Monitor    8.0   350.0          0
4     Chair    0.0     0.0       FREE

--- Column-Specific Fill ---
    Product  Stock   Price Promo_Code
0    Laptop   15.0  1200.0     SAVE10
1     Mouse   15.0    25.5        N/A
2  Keyboard   42.0   525.2    WELCOME
3   Monitor    8.0   350.0       NONE
4     Chair   15.0   525.2       FREE
```

---

## 4. Replacing Custom Placeholders with `.replace()`

Sometimes missing data is represented by custom string values (like `"N/A"`, `"null"`, or `-999`). Use `.replace()` to convert these to NumPy `NaN` values.

```python
# Replace custom string "N/A" with np.nan
df_replaced = df_missing.replace("N/A", np.nan)
print("--- Replaced Custom String with NaN ---")
print(df_replaced)
```

### Output

```text
--- Replaced Custom String with NaN ---
    Product  Stock   Price Promo_Code
0    Laptop   15.0  1200.0     SAVE10
1     Mouse    NaN    25.5        NaN
2  Keyboard   42.0     NaN    WELCOME
3   Monitor    8.0   350.0        NaN
4     Chair    NaN     NaN       FREE
```

---

## 5. Modern Nullable Data Types (`Int64`)

Compare the type conversion of standard integers vs nullable integers when a null value is introduced.

```python
# Standard integer list containing a None will force float conversion
s_float = pd.Series([1, 2, None])
print("Standard float conversion:\n", s_float)

# Specifying Pandas nullable integer type "Int64" preserves integers
s_int64 = pd.Series([1, 2, None], dtype="Int64")
print("\nPandas Nullable Int64:\n", s_int64)
```

### Output

```text
Standard float conversion:
 0    1.0
1    2.0
2    NaN
dtype: float64

Pandas Nullable Int64:
 0       1
1       2
2    <NA>
dtype: Int64
```
*Note: Pandas represents missing values in modern nullable types using `<NA>` instead of `NaN`.*

---

## Common Mistakes Students Make

- **Chaining `.dropna()` and losing indices**: When dropping records, students often forget that `.dropna()` removes the row index. Make sure to reset the index with `.reset_index(drop=True)` if sequential index integers are needed.
- **Accidental conversion to float**: Assigning a missing value to a standard int series (`s[0] = np.nan`) silently casts the whole series to float. Use `dtype="Int64"` to prevent this behavior.
- **Ignoring custom placeholders**: Relying on `.isna()` immediately after loading raw text data that uses string placeholders (like `"missing"`, `"null"`, or `"-1"`) will fail to detect them. Clean column values using `.replace()` first.
- **Applying `.fillna()` on multi-type DataFrames**: Filling the entire DataFrame with a single constant (like `df.fillna(0)`) replaces missing categorical strings with `0`, which can corrupt data quality. Use column-specific dictionaries instead.

---

## Best Practices

- Always run `df.isna().sum()` immediately after loading a dataset to inspect missing value distributions.
- Drop rows selectively using the `subset` parameter to avoid discarding valid records in other columns.
- When replacing missing numbers with summary statistics, use the median instead of the mean if the data contains extreme outliers.
- Prefer Pandas nullable types (`Int64`, `Float64`, `boolean`) when defining database-ready schemas.

---

## Worked Real-World Examples

### Worked Example 1: Cleaning Sensor Log Records

```python
import pandas as pd
import numpy as np

# 1. Initialize dataset tracking industrial sensor parameters
logs = pd.DataFrame({
    "Timestamp": pd.date_range("2026-06-01 12:00", periods=5, freq="min"),
    "Sensor_ID": ["A_10", "A_10", "A_10", "A_10", "A_10"],
    "Vibration": [0.15, np.nan, 0.18, np.nan, 0.22],
    "Status": ["Active", "Active", "Error", "Active", np.nan]
})

# 2. Keep records only if Status is present
logs_cleaned = logs.dropna(subset=["Status"]).copy()

# 3. Impute missing vibration readings using forward fill
logs_cleaned["Vibration"] = logs_cleaned["Vibration"].ffill()

print("--- Cleaned Sensor Logs ---")
print(logs_cleaned)
```

### Output

```text
--- Cleaned Sensor Logs ---
            Timestamp Sensor_ID  Vibration  Status
0 2026-06-01 12:00:00      A_10       0.15  Active
1 2026-06-01 12:01:00      A_10       0.15  Active
2 2026-06-01 12:02:00      A_10       0.18   Error
3 2026-06-01 12:03:00      A_10       0.18  Active
```

---

### Worked Example 2: E-commerce Customer Profile Completion

```python
import pandas as pd
import numpy as np

# Customer demographics
df_cust = pd.DataFrame({
    "CustID": [101, 102, 103, 104],
    "Age": [25, np.nan, 40, np.nan],
    "City": ["Delhi", np.nan, "Mumbai", "Delhi"],
    "Email": ["abc@test.com", "xyz@test.com", np.nan, "demo@test.com"]
})

# Calculate median age
median_age = df_cust["Age"].median()

# Define column fill values
fill_rules = {
    "Age": median_age,
    "City": "Unknown",
    "Email": "no-email@test.com"
}

df_conformed = df_cust.fillna(value=fill_rules)
print("--- Completed Customer Profiles ---")
print(df_conformed)
```

### Output

```text
--- Completed Customer Profiles ---
   CustID   Age     City              Email
0     101  25.0    Delhi       abc@test.com
1     102  32.5  Unknown       xyz@test.com
2     103  40.0   Mumbai  no-email@test.com
3     104  32.5    Delhi      demo@test.com
```

---

### Worked Example 3: Imputing Sales with Limit Restraints

```python
import pandas as pd
import numpy as np

s_sales = pd.Series([100.0, np.nan, np.nan, np.nan, 250.0])

# Impute missing sales using forward fill, limiting interpolation to 1 consecutive period
s_imputed = s_sales.ffill(limit=1)

print("--- Original Sales Series ---")
print(s_sales.values)
print("\n--- Imputed Sales (Limit=1) ---")
print(s_imputed.values)
```

### Output

```text
--- Original Sales Series ---
[100.  nan  nan  nan 250.]

--- Imputed Sales (Limit=1) ---
[100. 100.  nan  nan 250.]
```

---

## Practice Questions

1. Explain the differences between the `.isna()` and `.isnull()` methods in Pandas.
2. Why does Pandas traditionally convert integer columns containing missing values to float?
3. Write a command to count the number of missing values in each row of a DataFrame.
4. Explain the function of the `thresh` parameter inside the `.dropna()` method.
5. Create a Series with values `[10, None, 20]` using the Pandas nullable integer type.
6. What is the behavior of `.fillna(method='ffill')` when the first entry of the Series is missing?
7. Write a script to replace the integer placeholder `-999` with NaN across an entire DataFrame.
8. How does the `limit` parameter in `.fillna()` prevent over-imputing data?
9. Compare the memory storage footprint of a float64 Series containing `NaN`s versus an Int64 Series containing `<NA>`s.
10. Write a statement to drop columns that have more than 50% missing values.

---

## Mini Assignments

### Assignment 1: Student Attendance Sheet Completion
- Create an attendance tracker for 8 students containing `Student_ID`, `Days_Attended` (with nulls), and `Test_Score` (with nulls).
- Impute the missing values in `Days_Attended` with the column mode.
- Impute the missing values in `Test_Score` with the class median score.
- Confirm the types of both columns post-fill.

### Assignment 2: Industrial Logs Anomaly Purge
- Create a manufacturing logs dataset tracking: `Timestamp`, `Pressure` (with nulls), `Vibration` (with nulls), and `Status`.
- Drop any rows where both `Pressure` and `Vibration` are missing.
- For remaining rows, fill missing values in `Pressure` using backward filling.

### Assignment 3: Survey Response Standardisation
- Generate a customer survey dataset with columns `Response_ID`, `Rating` (containing strings `"N/A"`, `"none"`), and `Comments`.
- Replace the `"N/A"` and `"none"` placeholders with standard `NaN` values.
- Calculate the total count and percentage of missing rating entries.

---

## Interview-Oriented Questions

- **What is the difference between standard integer arrays and Pandas' nullable integer `Int64` type?**
  - *Answer*: Standard NumPy integer arrays do not support missing values, so introducing a null value forces the column type to cast to float64 (where NaN is represented as a float). The Pandas `Int64` type uses a separate boolean mask array under the hood to track missing elements (`<NA>`), allowing integers to be stored as integers without type conversion.
- **Under what circumstances should you choose median imputation over mean imputation?**
  - *Answer*: Median imputation is preferred when the data contains extreme outliers or is highly skewed. The mean is sensitive to extreme values, which can pull the imputed value away from the typical range of the data. The median represents the middle value of the distribution and is robust to outliers.
- **Explain the behavior of the `how='all'` parameter inside the `.dropna()` method.**
  - *Answer*: `how='all'` will drop a row or column only if all of its values are missing. If even a single value is present, the row/column is preserved. In contrast, `how='any'` (the default) drops the row/column if at least one value is missing.
- **How does standard aggregations (like `.sum()` or `.mean()`) handle missing values in Pandas?**
  - *Answer*: By default, statistical operations in Pandas ignore missing values (`skipna=True`). For example, `.mean()` calculates the average of only the non-null entries. If all values in the column are null, the operation returns `NaN`.
- **Explain the difference between `isna().sum()` and `isna().any()`.**
  - *Answer*: `isna().sum()` returns the total count of missing values in each column (or row). `isna().any()` returns a boolean Series indicating whether each column (or row) contains *at least one* missing value.

---

## Teaching Notes for This Chapter

- **Demonstrate Type Casting**: Show students how `int` columns change to `float` when a single `None` is introduced. Highlight how this can cause downstream issues with operations that expect integers.
- **Differentiate between dropna variants**: Use a small 3x3 DataFrame to illustrate the differences between `dropna()`, `dropna(axis=1)`, `dropna(how='all')`, and `dropna(subset=[...])`.
- **Emphasize Data Imputation Risks**: Explain that imputation is an estimation, and filling too many missing values can distort the statistical properties of the dataset.

---

## Chapter Wrap-up Concepts Students Must Master

- Use `.isna()` to check for missing values and `.isna().sum()` to count them.
- `.dropna()` removes rows or columns based on missing value presence.
- `.fillna()` replaces missing values with constants, calculated statistics, or interpolated values.
- Standard integer columns convert to float when missing values are introduced; use the Pandas nullable integer type `Int64` to prevent this.
- Replace custom placeholders (like `"N/A"`) using `.replace()` before running missing-data operations.
