# Data Types and Type Conversion (numeric, categorical, datetime)

## Lesson Overview

- This chapter explores data types in Pandas. We cover inspecting data types, casting columns using `.astype()`, using error-robust parsers (`pd.to_numeric()`, `pd.to_datetime()`), and optimizing memory using the `category` data type.
- Columns in raw datasets are often imported with incorrect types (e.g. numeric prices containing currency symbols imported as text strings). If columns are not typed correctly, mathematical operations, time-series plotting, and groupings will fail or produce incorrect results.
- We will cover type-safety checks, error coercion, and how to convert objects to numeric, datetime, and categorical types.
- Mastering type conversions helps reduce memory footprints and ensures compatibility with downstream machine learning and plotting libraries.

## Learning Objectives

- Inspect DataFrame column types using `df.dtypes` and summary information with `df.info()`.
- Cast columns between standard data types (e.g., float to integer, numeric to string) using `.astype()`.
- Apply `pd.to_numeric()` with `errors='coerce'` to parse mixed numeric columns safely.
- Convert text-based date logs to datetime objects using `pd.to_datetime()`.
- Transition repeated string columns into optimized categorical types to reduce memory usage.

---

## Pandas Data Types

Pandas maps columns to specific data types based on the values they contain:

| Data Type | Description | Python Equivalent |
| :--- | :--- | :--- |
| **`int64`** or **`Int64`** | Signed integers (64-bit). `Int64` is the nullable version. | `int` |
| **`float64`** | Floating-point numbers. | `float` |
| **`object`** | Text strings, mixed types, or python objects. | `str` or any python object |
| **`bool`** or **`boolean`** | Boolean (True/False). `boolean` is nullable. | `bool` |
| **`datetime64[ns]`** | Date and time values with nanosecond precision. | `datetime.datetime` |
| **`category`** | Fixed list of categorical text fields (highly optimized). | N/A |

### Setup for Demonstration

```python
import pandas as pd

# Raw dataset with inconsistent types
df_raw = pd.DataFrame({
    "Product": ["Laptop", "Mouse", "Keyboard", "Laptop"],
    "Price": ["1200.00", "25.50", "invalid_price", "1200.00"],
    "Launch_Date": ["2026-01-15", "2026-02-10", "2026-03-01", "2026-01-15"],
    "Units_Sold": [150.0, 420.0, 310.0, 150.0]
})

print("--- Raw Column Types ---")
print(df_raw.dtypes)
```

### Output

```text
--- Raw Column Types ---
Product         object
Price           object
Launch_Date     object
Units_Sold     float64
dtype: object
```

---

## 1. Explicit Casting with `.astype()`

The `.astype()` method converts column types explicitly. It raises an error if any values cannot be cast.

```python
# Convert Units_Sold from float64 to int64
df_cast = df_raw.copy()
df_cast["Units_Sold"] = df_cast["Units_Sold"].astype("int64")

# Convert Product names to strings (objects)
df_cast["Product"] = df_cast["Product"].astype("str")

print("--- Casted Columns ---")
print(df_cast[["Product", "Units_Sold"]].dtypes)
```

### Output

```text
--- Casted Columns ---
Product       object
Units_Sold     int64
dtype: object
```

---

## 2. Safe Parsing with `pd.to_numeric()`

If a column contains invalid characters (like `"invalid_price"`), `.astype('float')` raises a `ValueError`. Use `pd.to_numeric()` to handle errors:
- `errors='raise'`: Raise a traceback error (default).
- `errors='ignore'`: Return the original input without changes.
- `errors='coerce'`: Replace invalid values with `NaN`.

```python
# Attempting df_raw["Price"].astype(float) throws ValueError: could not convert string to float

# Safely parse numeric entries, coercing invalid values to NaN
df_cast["Price"] = pd.to_numeric(df_raw["Price"], errors="coerce")

print("--- Parsed Price Column ---")
print(df_cast["Price"])
print("New Type:", df_cast["Price"].dtype)
```

### Output

```text
--- Parsed Price Column ---
0    1200.0
1      25.5
2       NaN
3    1200.0
Name: Price, dtype: float64
New Type: float64
```

---

## 3. Parsing Datetimes with `pd.to_datetime()`

Convert text strings representing dates into actual datetime objects. This enables date-based filtering, sorting, and time-series aggregations.

```python
# Convert text strings to datetime64[ns]
df_cast["Launch_Date"] = pd.to_datetime(df_raw["Launch_Date"], format="%Y-%m-%d")

print("--- Parsed Launch Date ---")
print(df_cast["Launch_Date"])
print("New Type:", df_cast["Launch_Date"].dtype)
```

### Output

```text
--- Parsed Launch Date ---
0   2026-01-15
1   2026-02-10
2   2026-03-01
3   2026-01-15
Name: Launch_Date, dtype: datetime64[ns]
New Type: datetime64[ns]
```

---

## 4. Optimizing Memory with Categorical Data Types

Text columns with repeating values (like departments, country codes, or states) consume significant memory. The `category` type maps distinct strings to integer codes under the hood, reducing memory usage.

```python
# Generate large series of repeated categories
s_large_obj = pd.Series(["Tech", "Sales", "HR", "Tech"] * 10000)
s_large_cat = s_large_obj.astype("category")

print("--- Memory Footprint Comparison ---")
print("Object Series (Bytes):", s_large_obj.memory_usage(deep=True))
print("Category Series (Bytes):", s_large_cat.memory_usage(deep=True))
```

### Output

```text
--- Memory Footprint Comparison ---
Object Series (Bytes): 2520080
Category Series (Bytes): 40516
```
*Note: Converting repeated text columns to `category` can reduce memory footprints by 95% or more.*

---

## Common Mistakes Students Make

- **Using `.astype(int)` on columns containing nulls**: In standard Pandas, casting a column containing `NaN` to `int` raises a `ValueError: Cannot convert non-finite values (NA or inf) to integer`. Convert the column to the nullable integer type `'Int64'` (capital I) or fill missing values before casting.
- **Silently ignoring errors with `errors='ignore'`**: When using `pd.to_numeric(df['Col'], errors='ignore')`, if even a single row contains an invalid character, the entire column remains an `object` type. Always verify the resulting column dtype.
- **Mismatching datetime format strings**: Passing an incorrect format string (e.g. `%d-%m-%Y` for a `"2026-12-05"` date) raises a `ParserError`. Ensure the format string matches the date layout.
- **Converting low-cardinality text columns to category**: If a text column contains mostly unique values (e.g. `UserID` or `Email`), converting it to `category` actually increases memory usage because Pandas has to store a large lookup table of unique strings. Only use categories for low-cardinality columns.

---

## Best Practices

- Always run `df.info()` immediately after loading data to verify that numeric columns are not imported as objects.
- Use `pd.to_numeric(..., errors='coerce')` to parse columns containing currency symbols or punctuation, and fill the resulting NaNs as needed.
- Standardize on `datetime64[ns]` formats for all date and time columns to enable datetime operations.
- Convert text columns with repeating values (like states, departments, or ratings) to the `category` type to reduce memory usage.

---

## Worked Real-World Examples

### Worked Example 1: Sanitizing Mixed Transaction Logs

```python
import pandas as pd

# Ingested sales data
raw_sales = pd.DataFrame({
    "OrderID": [101, 102, 103, 104],
    "Price": ["$1,200.00", "$45.50", "Cancelled", "$85.00"],
    "Quantity": [2, 1, 0, 3]
})

# 1. Clean the Price column by removing currency symbols and commas
clean_price = raw_sales["Price"].str.replace("$", "", regex=False).str.replace(",", "", regex=False)

# 2. Parse price column, converting "Cancelled" to NaN
raw_sales["Price"] = pd.to_numeric(clean_price, errors="coerce")

# 3. Cast Quantity to nullable Int64
raw_sales["Quantity"] = raw_sales["Quantity"].astype("Int64")

print("--- Sanitized Sales Ledger ---")
print(raw_sales)
print("\nColumn Types:")
print(raw_sales.dtypes)
```

### Output

```text
--- Sanitized Sales Ledger ---
   OrderID   Price  Quantity
0      101  1200.0         2
1      102    45.5         1
2      103     NaN         0
3      104    85.0         3

Column Types:
OrderID       int64
Price       float64
Quantity      Int64
dtype: object
```

---

### Worked Example 2: Financial Sales Report Generation

```python
import pandas as pd

# Sales registrations
df_sales = pd.DataFrame({
    "Date": ["06/01/2026", "06/02/2026", "06/03/2026"],
    "Sales_Region": ["North", "North", "South"],
    "Revenue": [25000, 18000, 32000]
})

# 1. Parse date column to datetime64
df_sales["Date"] = pd.to_datetime(df_sales["Date"], format="%m/%d/%Y")

# 2. Convert Sales_Region to Category
df_sales["Sales_Region"] = df_sales["Sales_Region"].astype("category")

# 3. Filter data using datetime operations
june_second_onward = df_sales[df_sales["Date"] >= "2026-06-02"]

print("--- June 2nd Onward Sales ---")
print(june_second_onward)
```

### Output

```text
--- June 2nd Onward Sales ---
        Date Sales_Region  Revenue
1 2026-06-02        North    18000
2 2026-06-03        South    32000
```

---

### Worked Example 3: Parsing Custom Time Formats

```python
import pandas as pd

df_logs = pd.DataFrame({
    "System": ["S1", "S2"],
    "Log_Timestamp": ["2026-06-01T12:00:00.000Z", "2026-06-01T12:05:00.000Z"]
})

# Parse ISO-8601 UTC date formats
df_logs["Log_Timestamp"] = pd.to_datetime(df_logs["Log_Timestamp"], format="ISO8601")

print("--- Parsed Log Timestamps ---")
print(df_logs)
print("Log Column Type:", df_logs["Log_Timestamp"].dtype)
```

### Output

```text
--- Parsed Log Timestamps ---
  System             Log_Timestamp
0     S1 2026-06-01 12:00:00+00:00
1     S2 2026-06-01 12:05:00+00:00
Log Column Type: datetime64[ns, UTC]
```

---

## Practice Questions

1. Identify the 6 standard data types supported by the Pandas library.
2. Explain the difference in behavior between `.astype('float')` and `pd.to_numeric()`.
3. Write a command to convert a DataFrame column containing dates in the format `"31-12-2026"` to datetime.
4. Under what conditions does converting a string column to the `category` type increase memory usage?
5. Write a command to convert float column `df['Cost']` to the nullable integer type `Int64`.
6. What is the difference between `datetime64[ns]` and the standard Python `datetime` object?
7. Write a script to convert all boolean columns in a DataFrame to standard `int8` columns.
8. How does `errors='coerce'` handle invalid inputs when parsing data?
9. Compare the memory usage of an object Series versus a category Series for a column with 1,000,000 entries and only 3 unique values.
10. Describe how Pandas handles date parsing when timezone markers are present in the date string.

---

## Mini Assignments

### Assignment 1: Student Demographics Optimization
- Create a student register with 10 records containing: `Student_ID`, `City_Branch` (highly duplicated strings), `Registration_Fees` (e.g. `"$150"`, `"$200"`), and `Admission_Date` (strings like `"2026-06-01"`).
- Clean and convert the numeric fees to standard floats.
- Optimize the memory footprint of the branch locations using the category type.
- Convert the registration date to a datetime object.

### Assignment 2: Industrial Sensor Type Alignment
- Create a sensor log DataFrame containing `Sensor_Code`, `Temp_Reading` (floats with some nulls), and `Active_Status` (integers `1` and `0`).
- Cast `Active_Status` to a boolean type.
- Convert `Temp_Reading` to float32.
- Verify and print the memory footprint of the DataFrame before and after optimization.

### Assignment 3: Financial Transaction Date Parser
- Load a payment ledger containing columns: `InvoiceID`, `Amount`, and `Paid_At` (dates in the format `"01/Dec/2026 15:30:00"`).
- Parse the date column using `pd.to_datetime()` with a custom format string.
- Filter and output payments received after 12:00 PM.

---

## Interview-Oriented Questions

- **Why does converting a low-cardinality string column to a category data type save memory?**
  - *Answer*: When a column is cast to `category`, Pandas stores the unique strings in a lookup table (dictionary) and represents each row value using an integer code (typically 8-bit or 16-bit). This avoids storing the full string value in memory for every row, reducing the memory footprint for columns with duplicate strings.
- **Explain the difference between `errors='coerce'` and `errors='ignore'` inside `pd.to_numeric()`.**
  - *Answer*: `errors='coerce'` replaces invalid numeric strings (like `"cancelled"` or `"N/A"`) with `NaN`, allowing the column to be cast to `float64` or `Int64`. `errors='ignore'` returns the input series unchanged without throwing an error, meaning the column remains an `object` type if it contains even one invalid string.
- **What is the consequence of casting a column containing floats and missing values (`NaN`) to standard `int64`?**
  - *Answer*: Attempting to cast a column containing floating-point `NaN` values to standard `int64` raises a `ValueError: Cannot convert non-finite values (NA or inf) to integer`. To resolve this, use Pandas' nullable integer type `'Int64'` or fill missing values before casting.
- **How can we identify all columns of a specific data type in a DataFrame?**
  - *Answer*: Use the `df.select_dtypes(include=[type])` method. For example, `df.select_dtypes(include=['category'])` returns a subset DataFrame containing only categorical columns.
- **What data type does Pandas assign to columns containing mixed types (e.g., integers and strings)?**
  - *Answer*: Columns containing mixed types are assigned the `object` data type. Operations on these columns can be slower because Pandas must evaluate each entry dynamically.

---

## Teaching Notes for This Chapter

- **Demonstrate Memory Savings**: Show students the memory difference between `object` and `category` types using a large DataFrame in class.
- **Explain Datetime Formats**: Clearly explain format symbols like `%Y`, `%m`, `%d`, `%H`, `%M`, and `%S`, as students often confuse capitalization (e.g. `%m` for month versus `%M` for minute).
- **Highlight Nullable Integers**: Emphasize the difference between lowercase `int64` (non-nullable) and uppercase `Int64` (nullable) to prevent casting errors in class.

---

## Chapter Wrap-up Concepts Students Must Master

- Use `df.dtypes` to inspect column types and `df.info()` to view memory details.
- Cast columns explicitly using `.astype()`, and use nullable types (`Int64`, `Float64`, `boolean`) to preserve integers and booleans with missing values.
- Use `pd.to_numeric(..., errors='coerce')` to parse mixed numeric columns safely.
- Convert text dates to datetime objects using `pd.to_datetime()` to enable time-series calculations.
- Use categorical types (`category`) for string columns with repeating values to reduce memory footprints.
