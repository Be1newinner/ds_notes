# Boolean Indexing and Query-Based Filtering

## Lesson Overview

- This chapter provides an exhaustive deep dive into filtering and querying datasets in Pandas using boolean logic, index-level masks, and the high-level `.query()` evaluation engine.
- Real-world data is inherently noisy and massive. Analysts rarely process entire datasets at once; instead, they filter records based on specific criteria—such as isolating high-value customers, detecting sensor anomalies, or identifying transactions within a specific geographic area.
- Improper filtering syntax, such as omitting parenthesis during multi-condition combinations or using Python's logical `and`/`or` instead of Pandas bitwise `&`/`|`, leads to syntax errors or incorrect results.
- Mastering boolean masking and query-based filtering ensures clean, readable, high-performance data extraction, which is essential for feature engineering and advanced analysis.

## Learning Objectives

- Construct boolean masks to select rows matching specific scalar thresholds or category matches.
- Combine multiple filtering conditions using bitwise operators (`&`, `|`, `~`) and enforce execution order with parentheses.
- Leverage convenience filters such as `.isin()`, `.between()`, and missing-data selectors (`.isna()`, `.notna()`).
- Utilize the `.query()` method to write cleaner, string-based selection queries, and pass dynamic variables using the `@` prefix.
- Understand the performance differences between boolean indexing and the `.query()` engine on small versus large datasets.
- Avoid common logical pitfalls and debug filtering errors efficiently.

---

## The Mechanics of Boolean Indexing

Boolean indexing (also known as boolean masking) is the process of filtering a DataFrame or Series by passing a boolean array (True/False values) of the same length as the index. Rows corresponding to `True` are kept; rows corresponding to `False` are discarded.

When you apply a comparison operator directly to a Series, Pandas evaluates the comparison element-wise, returning a new Series of boolean values.

### Setup for Demonstration

```python
import pandas as pd

# Sample dataset representing e-commerce transactions
df_orders = pd.DataFrame({
    "OrderID": [1001, 1002, 1003, 1004, 1005],
    "Customer": ["Alice", "Bob", "Charlie", "David", "Eva"],
    "Category": ["Tech", "Home", "Tech", "Beauty", "Home"],
    "Price": [1200.50, 45.00, 850.00, 150.25, 30.00],
    "Quantity": [1, 3, 2, 1, 10],
    "Shipped": [True, False, True, True, False]
})

print("--- Master Orders Table ---")
print(df_orders)
```

### Output

```text
--- Master Orders Table ---
   OrderID  Customer Category    Price  Quantity  Shipped
0     1001     Alice     Tech  1200.50         1     True
1     1002       Bob     Home    45.00         3    False
2     1003   Charlie     Tech   850.00         2     True
3     1004     David   Beauty   150.25         1     True
4     1005       Eva     Home    30.00        10    False
```

---

### 1. Basic Single-Condition Filtering

To filter rows, create a boolean condition and pass it inside the indexer brackets `[]` or `.loc[]`.

```python
# Create boolean mask
tech_mask = df_orders["Category"] == "Tech"
print("--- Boolean Mask Series ---")
print(tech_mask)

# Apply mask to filter DataFrame
tech_orders = df_orders[tech_mask]
print("\n--- Filtered Tech Orders ---")
print(tech_orders)
```

### Output

```text
--- Boolean Mask Series ---
0     True
1    False
2     True
3    False
4    False
Name: Category, dtype: bool

--- Filtered Tech Orders ---
   OrderID  Customer Category   Price  Quantity  Shipped
0     1001     Alice     Tech  1200.5         1     True
2     1003   Charlie     Tech   850.0         2     True
```

---

### 2. Multi-Condition Filtering (Bitwise Operators)

To combine multiple conditions, you **must** use Pandas-specific bitwise operators:
- `&` (and)
- `|` (or)
- `~` (not)

Additionally, **each individual condition must be enclosed in parentheses `()`** to override Python's operator precedence.

```python
# Select orders where category is Tech AND price is greater than 1000
expensive_tech = df_orders[(df_orders["Category"] == "Tech") & (df_orders["Price"] > 1000)]
print("--- Expensive Tech Orders ---")
print(expensive_tech)

# Select orders where quantity is greater than 5 OR price is less than 50
bulk_or_cheap = df_orders[(df_orders["Quantity"] > 5) | (df_orders["Price"] < 50)]
print("\n--- Bulk or Cheap Orders ---")
print(bulk_or_cheap)
```

### Output

```text
--- Expensive Tech Orders ---
   OrderID  Customer Category   Price  Quantity  Shipped
0     1001     Alice     Tech  1200.5         1     True

--- Bulk or Cheap Orders ---
   OrderID  Customer Category  Price  Quantity  Shipped
1     1002       Bob     Home   45.0         3    False
4     1005       Eva     Home   30.0        10    False
```

---

## Essential Convenience Methods: `.isin()`, `.between()`, and Null Checkers

Writing multiple OR conditions can quickly make code unreadable. Pandas offers optimized convenience functions to streamline filtering logic.

### 1. `.isin()` for Multiple Values Matching

Instead of writing `(df["Col"] == "A") | (df["Col"] == "B")`, pass a list of values to `.isin()`.

```python
# Select orders belonging to Tech or Beauty categories
tech_beauty_orders = df_orders[df_orders["Category"].isin(["Tech", "Beauty"])]
print("--- Tech & Beauty Categories ---")
print(tech_beauty_orders)
```

### Output

```text
--- Tech & Beauty Categories ---
   OrderID  Customer Category    Price  Quantity  Shipped
0     1001     Alice     Tech  1200.50         1     True
2     1003   Charlie     Tech   850.00         2     True
3     1004     David   Beauty   150.25         1     True
```

### 2. `.between()` for Range Checkers

Instead of writing `(df["Col"] >= start) & (df["Col"] <= end)`, use `.between(start, end)`. By default, `.between` is inclusive of boundaries.

```python
# Select orders where price is between 100 and 900 inclusive
range_orders = df_orders[df_orders["Price"].between(100.00, 900.00)]
print("--- Orders Priced between 100 and 900 ---")
print(range_orders)
```

### Output

```text
--- Orders Priced between 100 and 900 ---
   OrderID  Customer Category   Price  Quantity  Shipped
2     1003   Charlie     Tech  850.00         2     True
3     1004     David   Beauty  150.25         1     True
```

### 3. `.isna()` and `.notna()` for Null Filtering

Check for missing values (represented as `NaN` or `None`) to clean datasets.

```python
import numpy as np

df_nan = pd.DataFrame({"Item": ["A", "B", "C"], "Weight": [1.5, np.nan, 3.4]})

# Keep rows where Weight is NOT null
cleaned_df = df_nan[df_nan["Weight"].notna()]
print("--- Excluded NaNs ---")
print(cleaned_df)
```

### Output

```text
--- Excluded NaNs ---
  Item  Weight
0    A     1.5
2    C     3.4
```

---

## High-Level Querying with `.query()`

The `.query()` method allows you to filter DataFrames using a clean string expression instead of writing verbose pandas syntax.

### Basic `.query()` Usage

```python
# Equivalent to: df_orders[(df_orders["Category"] == "Tech") & (df_orders["Price"] > 500)]
query_res = df_orders.query("Category == 'Tech' and Price > 500")
print("--- Query Results ---")
print(query_res)
```

### Output

```text
--- Query Results ---
   OrderID  Customer Category   Price  Quantity  Shipped
0     1001     Alice     Tech  1200.5         1     True
2     1003   Charlie     Tech   850.0         2     True
```

### Passing Dynamic External Variables with `@`

You can reference local variables inside the query string by prefixing them with the `@` symbol.

```python
min_price = 100
shipped_status = True

# Reference external variables using @
dynamic_res = df_orders.query("Price > @min_price and Shipped == @shipped_status")
print("--- Dynamic Query Results ---")
print(dynamic_res)
```

### Output

```text
--- Dynamic Query Results ---
   OrderID  Customer Category    Price  Quantity  Shipped
0     1001     Alice     Tech  1200.50         1     True
2     1003   Charlie     Tech   850.00         2     True
3     1004     David   Beauty   150.25         1     True
```

### Performance & Memory Efficiency: Boolean Indexing vs `.query()`

- **Boolean Indexing**: Faster for smaller datasets because it directly compiles to NumPy array expressions with minimal overhead. It evaluates in standard Python/NumPy execution space.
- **`.query()`**: Uses the **NumExpr** engine under the hood. It compiles the query string to high-speed virtual machine instructions, avoiding the creation of large intermediate arrays in memory.
- **Rule of Thumb**: For DataFrames with more than ~200,000 rows, `.query()` can be more memory efficient and faster. For small datasets, standard boolean indexing is typically faster.

---

## Common Mistakes Students Make

- **Forgetting Parentheses**: Writing `df[df['A'] > 5 & df['B'] < 10]` causes a traceback error (`TypeError` or logical compilation error). Because bitwise `&` has a higher operator precedence than comparison operators, Pandas attempts to evaluate `5 & df['B']` first. Always write `df[(df['A'] > 5) & (df['B'] < 10)]`.
- **Using logical `and`, `or`, `not`**: Writing `df[(df['A'] > 5) and (df['B'] < 10)]` will raise a `ValueError` stating "The truth value of a Series is ambiguous." Always use `&`, `|`, and `~` for element-wise array logic.
- **Modifying copy instead of source**: Filtering a dataframe and then assigning values (e.g. `df[df['A'] > 5]['B'] = 10`) triggers the `SettingWithCopyWarning` and does not save back to the original DataFrame. Use `.loc` instead: `df.loc[df['A'] > 5, 'B'] = 10`.
- **Querying Column Names with Spaces**: If a column has spaces (e.g. `Total Price`), writing `df.query("Total Price > 100")` fails. Use backticks inside the query string: `df.query("`Total Price` > 100")`.

---

## Best Practices

- Standardize on `df.loc[mask, columns]` when filtering and updating data in a single step to ensure warning-free, thread-safe memory writes.
- Prefer `.isin()` instead of multiple chained OR conditions to keep syntax clean and maintainable.
- Use `.query()` to improve code readability, especially when sharing analysis with non-technical stakeholders or writing complex multi-variable filters.
- Leverage the inplace parameter `inplace=False` by default; avoid `inplace=True` unless memory constraints demand it, as it reduces method chaining capabilities.

---

## Worked Real-World Examples

### Worked Example 1: E-commerce Order Filtering

```python
import pandas as pd

# 1. Initialize dataset tracking e-commerce transactions
transactions = pd.DataFrame({
    "TxID": ["T10", "T11", "T12", "T13", "T14", "T15"],
    "Customer": ["Rohan", "Pooja", "Vikram", "Neha", "Amit", "Kiran"],
    "Payment_Method": ["Card", "UPI", "NetBanking", "UPI", "Card", "UPI"],
    "Amount": [1500, 450, 12000, 95, 3200, 150],
    "Discount_Applied": [True, False, True, False, False, True]
})

# 2. Filter high-value UPI payments or Card payments with discount applied
target_mask = (
    ((transactions["Payment_Method"] == "UPI") & (transactions["Amount"] > 200)) |
    ((transactions["Payment_Method"] == "Card") & (transactions["Discount_Applied"] == True))
)

filtered_tx = transactions[target_mask]
print("--- Filtered Transaction Register ---")
print(filtered_tx)
```

### Output

```text
--- Filtered Transaction Register ---
  TxID Customer Payment_Method  Amount  Discount_Applied
0  T10    Rohan             Card    1500              True
1  T11    Pooja              UPI     450             False
```

---

### Worked Example 2: HR Audit and Performance Grading

```python
import pandas as pd

# Employees dataset
df_hr = pd.DataFrame({
    "EmpID": ["E01", "E02", "E03", "E04", "E05"],
    "Name": ["Aarav", "Sunita", "Rajesh", "Meera", "Kabir"],
    "Rating": [4.2, 3.8, 4.9, 2.5, 4.5],
    "Experience_Years": [3, 6, 8, 2, 5],
    "Salary": [50000, 75000, 120000, 42000, 90000]
})

# Filter employees eligible for senior leadership bonus
# Criteria: Rating >= 4.0 AND Experience_Years >= 5 AND Salary < 100000
query_str = "Rating >= 4.0 and Experience_Years >= 5 and Salary < 100000"
eligible_staff = df_hr.query(query_str)

print("--- Leadership Candidates ---")
print(eligible_staff)
```

### Output

```text
--- Leadership Candidates ---
  EmpID   Name  Rating  Experience_Years  Salary
4   E05  Kabir     4.5                 5   90000
```

---

### Worked Example 3: IoT Telemetry Anomaly Detection

```python
import pandas as pd
import numpy as np

# Generate system sensor metrics
df_sensors = pd.DataFrame({
    "SensorID": ["S_01", "S_02", "S_03", "S_04", "S_05"],
    "Temperature": [25.4, np.nan, 42.1, -10.5, 85.0],
    "Humidity": [45.2, 55.0, np.nan, 12.0, 92.4]
})

# Detect anomalous records: Temperature is missing OR Humidity is missing OR Temperature is outside normal bounds (0 to 50 degrees)
anomalies = df_sensors[
    df_sensors["Temperature"].isna() | 
    df_sensors["Humidity"].isna() | 
    ~df_sensors["Temperature"].between(0.0, 50.0)
]

print("--- Telemetry Anomalies ---")
print(anomalies)
```

### Output

```text
--- Telemetry Anomalies ---
  SensorID  Temperature  Humidity
1     S_02          NaN      55.0
2     S_03         42.1       NaN
3     S_04        -10.5      12.0
4     S_05         85.0      92.4
```

---

## Practice Questions

1. Explain the logical reason why Python's native `and` operator raises a `ValueError` when applied between two Pandas Series.
2. Write a boolean mask expression that selects values in a Series `s` that are strictly greater than 10 and strictly less than 20 without using `.between()`.
3. Re-write the condition from Question 2 utilizing the `.between()` method with custom boundary exclusion.
4. Compose a `.loc[]` statement that filters rows where `df['Status']` is "Active" and updates `df['Approved']` to `True`.
5. How do you query a column name that contains spaces (e.g., `Daily Sales`) using the `.query()` method?
6. Write a query string that references a locally defined threshold list `allowed_codes = [101, 102, 105]` to filter a column named `Code`.
7. What are the key performance benefits of the `NumExpr` engine inside `.query()` compared to standard Python execution?
8. Compose a statement using `.isin()` to filter a DataFrame `df` where the `Region` column matches either "North", "East", or "West".
9. Show how the negation operator `~` is used to invert a complex multi-condition boolean mask.
10. Describe how Pandas handles missing (`NaN`) values when evaluating a boolean comparison like `df['Age'] > 25`.

---

## Mini Assignments

### Assignment 1: Customer Churn Risk Classification
- Create a DataFrame containing 10 customer records with columns: `CustomerID`, `Monthly_Usage_Hours`, `Support_Tickets`, and `Payment_Delay_Days`.
- Using boolean indexing, extract a subset of customers flagged as "High Risk": `Support_Tickets > 5` AND `Payment_Delay_Days > 10`.
- Using `.loc[]`, add a new column named `Risk_Tier` and set it to "Critical" for high-risk customers, and "Standard" for others.
- Print the final DataFrame and confirm that no warning flags were raised.

### Assignment 2: Warehouse Stock Alert Query
- Create a warehouse inventory dataset with 8 items tracking `Item_Name`, `Stock_Count`, `Reorder_Level`, and `Lead_Time_Days`.
- Write a single dynamic `.query()` statement referencing local variables `min_lead_time = 5` and standard threshold logic where `Stock_Count <= Reorder_Level` to trigger restocking alerts.
- Export the filtered alert table and display it.

### Assignment 3: Medical Log Audit
- Load a patient vitals dataframe containing some null values in `Heart_Rate` and `Body_Temp`.
- Filter out records where both `Heart_Rate` and `Body_Temp` are present, and temperature values are within standard ranges (`36.0` to `38.0` inclusive).
- Count the number of discarded incomplete records.

---

## Interview-Oriented Questions

- **Why is it mandatory to use bitwise operators (`&`, `|`, `~`) instead of logical ones (`and`, `or`, `not`) in Pandas filtering?**
  - *Answer*: Logical operators like `and` evaluate the truth value of the entire object as a single boolean, which fails because a Series contains multiple elements (raising "truth value of a Series is ambiguous"). Bitwise operators are overloaded in Pandas to evaluate element-wise comparisons, outputting an array of individual boolean matches.
- **Explain the purpose and mechanics of the `@` prefix inside the `.query()` method.**
  - *Answer*: The `@` prefix tells the query engine to look up the subsequent name in the local environment namespace instead of evaluating it as a DataFrame column name. This allows dynamic variable injection inside query strings.
- **Under what scenarios would you choose standard boolean indexing over the `.query()` method?**
  - *Answer*: Standard boolean indexing is faster and simpler for small datasets because it avoids the overhead of parsing and compiling query strings. It is also preferred when you need to use functions or complex masking operations that aren't supported in string syntax.
- **How does the operator precedence rules impact how you write multi-condition boolean filters in Pandas?**
  - *Answer*: Bitwise operators (`&`, `|`) have higher precedence than comparison operators (`>`, `<`). Without parentheses, Python evaluates comparisons out of order (e.g. `x > 5 & y < 10` parses as `5 & y` first), resulting in a `TypeError`. Parentheses ensure comparison operations are resolved before bitwise operations.
- **Does using `.query()` modify the original DataFrame, and how can we manage return states?**
  - *Answer*: By default, `.query()` does not modify the original DataFrame; it returns a new filtered copy. It supports an `inplace=True` parameter to modify the DataFrame directly, though standard practice avoids `inplace` to preserve method chaining.

---

## Teaching Notes for This Chapter

- **Deconstruct Operator Precedence**: Spend 5 minutes showing students the traceback error when parentheses are omitted from `df['A'] > 5 & df['B'] < 10`. Point out the exact line where python fails to evaluate the bitwise logic.
- **Demonstrate string parsing inside `.query()`**: Clearly illustrate how quote styles must alternate, e.g., double quotes outside, single quotes inside `"Category == 'Tech'"`.
- **Explain NumExpr engine visualizer**: Draw how large intermediate copies are avoided inside the NumExpr engine, keeping memory footprint low.
- **Reinforce missing data handling**: Show that operations like `NaN > 5` return `False`, which can lead to data exclusion if not handled explicitly.

---

## Chapter Wrap-up Concepts Students Must Master

- Boolean filtering extracts rows by passing a series of logical matches matching the index footprint.
- Combining conditions requires bitwise operators (`&`, `|`, `~`) enclosed in parentheses `()`.
- Specialized convenience operators like `.isin()`, `.between()`, `.isna()`, and `.notna()` simplify logical declarations.
- The `.query()` method allows clean string-based filtering and dynamic parameter evaluation with the `@` prefix.
- Use boolean indexing for small datasets/complex operations, and `.query()` for cleaner syntax and memory savings on massive datasets.
