# Function Application with apply, map, applymap, and pipe

## Lesson Overview

- This chapter covers custom function application in Pandas. We explore mapping values with `.map()`, applying calculations with `.apply()`, element-wise transformations with `df.map()` (formerly `.applymap()`), and structuring operations using `.pipe()`.
- Standard Pandas functions cover many common tasks, but custom business logic (e.g. classifying text comments, calculating tax tiers, or applying multi-column algorithms) requires custom functions.
- We will cover `.map()`, `.apply()`, `df.map()`, and `.pipe()`, and analyze the performance differences between custom function loops and vectorized operations.
- Mastering these function tools allows you to write clean, maintainable, and chainable data pipelines.

## Learning Objectives

- Map Series elements to new values using dictionaries or lookup functions with `.map()`.
- Apply custom functions along DataFrame rows or columns using `.apply()`.
- Perform element-wise transformations across entire DataFrames using `df.map()` (formerly `applymap()`).
- Structure clean, readable method-chaining pipelines using `.pipe()`.
- Compare the execution speeds of vectorized operations versus custom `.apply()` loops.

---

## 1. Mapping Elements with `.map()` (Series Only)

The `.map()` method is a **Series-only** tool. It maps each value in a Series to a new value using a dictionary lookup, another Series, or a custom single-argument function.

### Setup for Demonstration

```python
import pandas as pd

# Customer registry
df_users = pd.DataFrame({
    "Name": ["Aarav", "Neha", "Vikram"],
    "Status_Code": [1, 2, 9],
    "Score": [88.5, 94.0, 52.0]
})

print("--- Master Users Table ---")
print(df_users)
```

### Output

```text
--- Master Users Table ---
     Name  Status_Code  Score
0   Aarav            1   88.5
1    Neha            2   94.0
2  Vikram            9   52.0
```

---

### Applying `.map()`

```python
# Dictionary mapping status codes to descriptive labels
status_map = {1: "Active", 2: "Suspended", 3: "Inactive"}

# Map values, filling unmapped codes (like 9) with NaN
df_users["Status_Label"] = df_users["Status_Code"].map(status_map)

# Map values using a function
df_users["Score_Formatted"] = df_users["Score"].map(lambda x: f"{x:.1f}%")

print("--- Mapped Users DataFrame ---")
print(df_users)
```

### Output

```text
--- Mapped Users DataFrame ---
     Name  Status_Code  Score Status_Label Score_Formatted
0   Aarav            1   88.5       Active           88.5%
1    Neha            2   94.0    Suspended           94.0%
2  Vikram            9   52.0          NaN           52.0%
```
*Note: Value 9 was not in the mapping dictionary, so it was converted to `NaN`.*

---

## 2. Dynamic Processing with `.apply()` (Series & DataFrames)

The `.apply()` method applies a function along an axis of a DataFrame or element-wise across a Series.

### Applying to a Series
Applies a function to each individual element of the Series.

```python
# Grade scores based on thresholds
def grade_score(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    return "C"

df_users["Grade"] = df_users["Score"].apply(grade_score)
print("--- Graded Series ---")
print(df_users[["Name", "Score", "Grade"]])
```

### Output

```text
--- Graded Series ---
     Name  Score Grade
0   Aarav   88.5     B
1    Neha   94.0     A
2  Vikram   52.0     C
```

---

### Applying to a DataFrame (Row-wise vs Column-wise)
- **`axis=0`** (or `axis='index'`): Passes each column as a Series to the function (default).
- **`axis=1`** (or `axis='columns'`): Passes each row as a Series to the function, enabling multi-column calculations.

```python
df_metrics = pd.DataFrame({
    "Q1": [12000, 8000],
    "Q2": [14000, 9500]
}, index=["Store_A", "Store_B"])

# 1. Column-wise aggregation (axis=0): Calculate the range (max - min) of each column
col_ranges = df_metrics.apply(lambda x: x.max() - x.min(), axis=0)
print("Column Ranges:\n", col_ranges)

# 2. Row-wise aggregation (axis=1): Calculate total revenue for each row
row_totals = df_metrics.apply(lambda x: x["Q1"] + x["Q2"], axis=1)
print("\nRow Totals:\n", row_totals)
```

### Output

```text
Column Ranges:
 Q1    4000
Q2    4500
dtype: int64

Row Totals:
 Store_A    26000
Store_B    17500
dtype: int64
```

---

## 3. Element-wise DataFrame Transformations with `df.map()`

To apply a function to every individual cell in a DataFrame, use `df.map()`.
- **Note**: In older versions of Pandas, this method was named **`.applymap()`**. Pandas 2.1.0 deprecated `.applymap()` in favor of **`.map()`** on DataFrames, unifying the element-wise mapping name.

```python
# Format all numeric cells in the DataFrame
formatted_metrics = df_metrics.map(lambda x: f"${x:,.2f}")

print("--- Formatted Metrics Grid ---")
print(formatted_metrics)
```

### Output

```text
--- Formatted Metrics Grid ---
                  Q1          Q2
Store_A  $12,000.00  $14,000.00
Store_B   $8,000.00   $9,500.00
```

---

## 4. Structured Pipelines with `.pipe()`

The `.pipe()` method allows you to apply functions that accept and return DataFrames, enabling clean method chaining without nesting function calls.

```python
# Define pipe stages
def add_tax(df, rate=0.05):
    df["Sales_Tax"] = df["Q1"] * rate
    return df

def add_total(df):
    df["Total"] = df["Q1"] + df["Q2"] + df["Sales_Tax"]
    return df

# Execute the pipeline using .pipe()
pipeline_res = df_metrics.copy().pipe(add_tax, rate=0.10).pipe(add_total)

print("--- Pipeline Results ---")
print(pipeline_res)
```

### Output

```text
--- Pipeline Results ---
            Q1    Q2  Sales_Tax    Total
Store_A  12000  14000     1200.0  27200.0
Store_B   8000   9500      800.0  18300.0
```

---

## Performance Implications: Vectorization vs `.apply()`

Custom function applications using `.apply()` or `.map()` are **slow** because they run standard Python loops over the rows under the hood. Wherever possible, use Pandas' built-in vectorized operations, which run in optimized C code.

```python
import numpy as np
import time

# Create a large DataFrame with 1,000,000 rows
df_large = pd.DataFrame({"Val": np.random.rand(1000000)})

# Measure speed of .apply()
start = time.time()
res_apply = df_large["Val"].apply(lambda x: x * 2)
time_apply = time.time() - start

# Measure speed of Vectorized operation
start = time.time()
res_vectorized = df_large["Val"] * 2
time_vectorized = time.time() - start

print(f".apply() Loop Time: {time_apply:.5f} seconds")
print(fVectorized Time: {time_vectorized:.5f} seconds")
print(f"Vectorization Speedup: {time_apply / time_vectorized:.1f}x faster")
```

### Output

```text
.apply() Loop Time: 0.12500 seconds
Vectorized Time: 0.00150 seconds
Vectorization Speedup: 83.3x faster
```

---

## Common Mistakes Students Make

- **Using `.apply()` when a vectorized operation exists**: Writing `df['Col'].apply(lambda x: x + 5)` instead of the vectorized expression `df['Col'] + 5` slows down execution. Always look for a vectorized alternative before writing `.apply()`.
- **Confusing `.map()` and `.apply()`**: `.map()` is a Series-only method and is optimized for dictionary lookups. Calling `df.map()` works as an element-wise DataFrame mapping (formerly `.applymap()`), while `df.apply()` operates along columns or rows.
- **Forgetting `axis=1` for row-wise processing**: When attempting to compute a value using multiple columns of a row, students often write `df.apply(lambda r: r['A'] + r['B'])`, forgetting to specify `axis=1`. This raises a `KeyError` because Pandas defaults to `axis=0` (column-wise).
- **Silent type casting**: When using `.apply()` on rows (`axis=1`), Pandas passes each row as a Series to the custom function. If the row contains mixed data types (e.g. integers, floats, and strings), Pandas casts all values in the Series to the lowest common type (typically `object`), which can cause type errors inside the function.

---

## Best Practices

- Prioritize vectorized operations over `.apply()` loops to optimize performance.
- Use `.map()` with a dictionary lookup when replacing category codes or labels in a Series.
- Specify `axis=1` explicitly when writing custom multi-column formulas inside `.apply()`.
- Use `.pipe()` to organize data cleaning stages into readable, chainable pipelines.
- Standardize on `df.map()` instead of the deprecated `df.applymap()` in modern Pandas projects.

---

## Worked Real-World Examples

### Worked Example 1: Multi-Tier Tax Calculation

```python
import pandas as pd

# Employee salary register
salaries = pd.DataFrame({
    "Name": ["Aarav", "Neha", "Vikram", "Pooja"],
    "Salary": [45000, 85000, 120000, 30000]
})

# Multi-tier tax function
def calculate_tax(row):
    salary = row["Salary"]
    if salary > 100000:
        return salary * 0.30
    elif salary > 50000:
        return salary * 0.20
    return salary * 0.10

# Calculate tax row-wise
salaries["Tax_Amt"] = salaries.apply(calculate_tax, axis=1)

print("--- Tax Ledger ---")
print(salaries)
```

### Output

```text
--- Tax Ledger ---
     Name  Salary  Tax_Amt
0   Aarav   45000   4500.0
1    Neha   85000  17000.0
2  Vikram  120000  36000.0
3   Pooja   30000   3000.0
```

---

### Worked Example 2: Clean method-chaining ETL Pipeline

```python
import pandas as pd

# Messy client database
df_clients = pd.DataFrame({
    "Full_Name": [" Aarav Patel ", "neha sharma", " vikram rathore "],
    "Age": [25, 30, 42],
    "Signup_Channel": ["web", "app", "web"]
})

# Pipe stages
def clean_names(df):
    df["Full_Name"] = df["Full_Name"].str.strip().str.title()
    return df

def uppercase_channels(df):
    df["Signup_Channel"] = df["Signup_Channel"].str.upper()
    return df

def classify_age(df):
    df["Age_Group"] = df["Age"].apply(lambda x: "Senior" if x > 40 else "Standard")
    return df

# Chain pipeline stages
cleaned_crm = (
    df_clients.copy()
    .pipe(clean_names)
    .pipe(uppercase_channels)
    .pipe(classify_age)
)

print("--- Cleaned CRM Pipeline ---")
print(cleaned_crm)
```

### Output

```text
--- Cleaned CRM Pipeline ---
        Full_Name  Age Signup_Channel Age_Group
0     Aarav Patel   25            WEB  Standard
1     Neha Sharma   30            APP  Standard
2  Vikram Rathore   42            WEB    Senior
```

---

### Worked Example 3: Formatting Financial Grids

```python
import pandas as pd

df_financials = pd.DataFrame({
    "Gross_Rev": [120000, 85000],
    "Net_Profit": [25000, 12000]
}, index=["Store_A", "Store_B"])

# Format entire DataFrame to currency string using df.map
df_formatted = df_financials.map(lambda x: f"${x:,.0f}")

print("--- Formatted Financial Report ---")
print(df_formatted)
```

### Output

```text
--- Formatted Financial Report ---
          Gross_Rev Net_Profit
Store_A  $120,000      $25,000
Store_B   $85,000      $12,000
```

---

## Practice Questions

1. Explain the difference in usage between `.map()`, `.apply()`, and `df.map()` (formerly `.applymap()`).
2. Write a command to map Series `s` values using a dictionary `d`, keeping any unmapped values unchanged.
3. Why are vectorized operations in Pandas faster than custom functions evaluated with `.apply()`?
4. Write a command to apply a function that squares each value in all columns of a DataFrame.
5. Explain how to use `.apply(..., axis=1)` to perform row-wise operations on a DataFrame.
6. Write a command to calculate the range (max - min) for each column in a DataFrame using `.apply()`.
7. What error occurs when you call `.map()` on a DataFrame in older versions of Pandas, and how has this changed?
8. Write a script that uses `.pipe()` to clean column names by stripping spaces and replacing them with underscores.
9. Explain how Pandas passes row data to a custom function when running `.apply(..., axis=1)`.
10. Describe how to apply a function that returns a Series with multiple values to a DataFrame column, expanding the results into new columns.

---

## Mini Assignments

### Assignment 1: Corporate Commission Calculation
- Create a sales DataFrame containing `Salesperson`, `Total_Sales`, and `Region`.
- Write a custom function that calculates a commission amount: if Region is `"North"`, commission is 10%; for other regions, commission is 8%.
- Apply the commission calculation row-wise and display the results.

### Assignment 2: Text Review Pipeline
- Create a CRM DataFrame containing customer comments.
- Use `.pipe()` to chain cleaning steps: strip whitespaces, convert to lowercase, and add a boolean flag `Is_Spam` that checks if the comment contains the word `"free"`.

### Assignment 3: Data Quality Flagging
- Create a manufacturing sensor DataFrame with columns: `S1_Vib`, `S2_Temp`, and `S3_Pres`.
- Write a custom threshold mapping function.
- Apply the mapping to the entire DataFrame using `df.map()` to flag any values exceeding specific limits with the string `"WARN"`.

---

## Interview-Oriented Questions

- **Why is using `.apply()` with a custom Lambda function slower than using vectorized arithmetic operators?**
  - *Answer*: Vectorized operations run compiled C code under the hood, performing array-level operations in a single step. `.apply()` runs a standard Python loop over the rows, passing each element or row to the custom function one by one. This introduces significant Python interpreter overhead and loops.
- **Explain the transition from `.applymap()` to `.map()` on DataFrames in recent Pandas versions.**
  - *Answer*: In Pandas 2.1.0, `.applymap()` was deprecated and replaced by `.map()` on DataFrame objects. This change unifies the method name for element-wise mapping across both Series and DataFrame structures, making the API cleaner and more consistent.
- **How does the `.pipe()` method improve readability in data cleaning pipelines?**
  - *Answer*: `.pipe()` allows you to apply functions that accept and return DataFrames in a chainable sequence (e.g. `df.pipe(clean).pipe(format)`). This avoids nested function calls (e.g. `format(clean(df))`) and prevents creating unnecessary temporary variables.
- **What is the consequence of omitting the `axis=1` parameter when calculating a metric based on multiple columns in `.apply()`?**
  - *Answer*: By default, `.apply()` uses `axis=0` (column-wise), which passes each column as a Series to the function. If your function expects a row Series with column keys, it will look for those keys in the columns index, raising a `KeyError` or a dimension mismatch error.
- **Does `.map()` on a Series preserve unmapped values, and how can we customize this behavior?**
  - *Answer*: By default, `.map()` replaces unmapped values with `NaN`. To preserve the original values, use `.replace()` with a dictionary lookup, or use `.map(lambda x: status_map.get(x, x))` to return the original value if it is missing from the dictionary.

---

## Teaching Notes for This Chapter

- **Deconstruct Loop Overhead**: Run a live code cell in class comparing the speed of `.apply()` versus a vectorized operation on a large DataFrame to illustrate performance differences.
- **Visualize Axis Directions**: Draw horizontal and vertical arrows on a DataFrame grid to illustrate how `axis=0` and `axis=1` work in `.apply()`.
- **Highlight Method Chaining**: Show how `.pipe()` makes code cleaner and easier to read compared to nested function calls.

---

## Chapter Wrap-up Concepts Students Must Master

- Vectorized operations are faster than `.apply()`; only use `.apply()` for custom logic that cannot be vectorized.
- Use `.map()` on a Series for dictionary-based value replacements.
- Use `.apply(..., axis=1)` to perform row-wise calculations that use multiple columns.
- Use `df.map()` (formerly `.applymap()`) for element-wise DataFrame transformations.
- Use `.pipe()` to organize data cleaning steps into chainable pipelines.
