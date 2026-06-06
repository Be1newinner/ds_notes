# Merging, Joining, and Concatenating DataFrames

## Lesson Overview

- This chapter explores combining and aligning datasets in Pandas using merging, joining, and concatenation.
- Data is rarely stored in a single table. To build comprehensive analytical pipelines, you must combine records from disparate sources—such as merging client signup lists with transactions, joining regional logs by timestamps, or stacking monthly sales tables.
- We will cover database-style joins using `pd.merge()`, index-based alignment using `df.join()`, stacking tables along both axes using `pd.concat()`, and using flags like `indicator=True` to track merge performance.
- Mastering these merge techniques prevents alignment failures and ensures data integrity when consolidating databases.

## Learning Objectives

- Execute database-style joins (inner, outer, left, right, cross) using `pd.merge()`.
- Resolve column conflicts and mismatched keys using `left_on`, `right_on`, and custom `suffixes`.
- Combine DataFrames by index labels using `.join()`.
- Stack datasets horizontally or vertically using `pd.concat()`, managing index resets with `ignore_index`.
- Track source records and audit join performance using the `indicator=True` flag.

---

## Database-Style Merges with `pd.merge()`

`pd.merge()` connects rows in DataFrames based on one or more keys, similar to SQL joins.

### Join Types (`how`)
- **`inner`**: Keep only keys present in both DataFrames (default).
- **`outer`**: Keep all keys from both DataFrames, filling missing entries with `NaN`.
- **`left`**: Keep all keys from the left DataFrame.
- **`right`**: Keep all keys from the right DataFrame.
- **`cross`**: Create a Cartesian product of both DataFrames.

### Setup for Demonstration

```python
import pandas as pd

# Customer registry
df_customers = pd.DataFrame({
    "CustomerID": [101, 102, 103, 104],
    "Name": ["Aarav", "Neha", "Vikram", "Pooja"],
    "City": ["Mumbai", "Delhi", "Mumbai", "Bangalore"]
})

# Orders registry
df_orders = pd.DataFrame({
    "OrderID": [5001, 5002, 5003],
    "Cust_ID": [101, 102, 105],
    "Amount": [2500, 1500, 3200]
})

print("--- Customers Table ---")
print(df_customers)
print("\n--- Orders Table ---")
print(df_orders)
```

### Output

```text
--- Customers Table ---
   CustomerID    Name       City
0         101   Aarav     Mumbai
1         102    Neha      Delhi
2         103  Vikram     Mumbai
3         104   Pooja  Bangalore

--- Orders Table ---
   OrderID  Cust_ID  Amount
0     5001      101    2500
1     5002      102    1500
2     5003      105    3200
```

---

### 1. Left Join with Mismatched Column Keys

Since the merge key is named `CustomerID` in the left table and `Cust_ID` in the right table, we use `left_on` and `right_on` parameters.

```python
# Left join to retain all customer profiles
customer_orders = pd.merge(
    df_customers,
    df_orders,
    left_on="CustomerID",
    right_on="Cust_ID",
    how="left"
)

print("--- Left Joined Customer Orders ---")
print(customer_orders)
```

### Output

```text
--- Left Joined Customer Orders ---
   CustomerID    Name       City  OrderID  Cust_ID  Amount
0         101   Aarav     Mumbai   5001.0    101.0  2500.0
1         102    Neha      Delhi   5002.0    102.0  1500.0
2         103  Vikram     Mumbai      NaN      NaN     NaN
3         104   Pooja  Bangalore      NaN      NaN     NaN
```
*Note: Customers 103 and 104 did not place orders, so their order details are filled with `NaN`.*

---

### 2. Merging with Suffixes and Indicators

If both DataFrames share columns that are not merge keys, Pandas appends suffixes (default: `_x`, `_y`) to differentiate them. Set `indicator=True` to add a column named `_merge` that tracks the source of each row.

```python
# Add a conflicting column 'City' to orders
df_orders["City"] = ["Mumbai", "Kolkata", "Delhi"]

# Merge with custom suffixes and indicator column
audit_merge = pd.merge(
    df_customers,
    df_orders,
    left_on="CustomerID",
    right_on="Cust_ID",
    how="outer",
    suffixes=("_cust", "_order"),
    indicator=True
)

print("--- Outer Joined Table with Audit ---")
print(audit_merge)
```

### Output

```text
--- Outer Joined Table with Audit ---
   CustomerID    Name  City_cust  OrderID  Cust_ID  Amount City_order      _merge
0       101.0   Aarav     Mumbai   5001.0    101.0  2500.0     Mumbai   both
1       102.0    Neha      Delhi   5002.0    102.0  1500.0    Kolkata   both
2       103.0  Vikram     Mumbai      NaN      NaN     NaN        NaN   left_only
3       104.0   Pooja  Bangalore      NaN      NaN     NaN        NaN   left_only
4         NaN     NaN        NaN   5003.0    105.0  3200.0      Delhi  right_only
```

---

## Index-Based Joins with `.join()`

The `.join()` method is a convenience function for combining DataFrames by their **index labels**. It defaults to a left join.

```python
# Create DataFrames with matching indexes
df_prices = pd.DataFrame({"Price": [100, 200]}, index=["P1", "P2"])
df_stocks = pd.DataFrame({"Stock": [15, 42]}, index=["P2", "P3"])

# Join tables by index
index_joined = df_prices.join(df_stocks, how="outer")

print("--- Index-Based Joined Table ---")
print(index_joined)
```

### Output

```text
--- Index-Based Joined Table ---
    Price  Stock
P1  100.0    NaN
P2  200.0   15.0
P3    NaN   42.0
```

---

## Concatenating Data along Axes with `pd.concat()`

`pd.concat()` stacks Series or DataFrames along an axis:
- `axis=0` (default): Stacks rows vertically.
- `axis=1`: Stacks columns horizontally.

```python
# Weekly transaction segments
df_week1 = pd.DataFrame({"TxID": ["T1", "T2"], "Amt": [150, 200]})
df_week2 = pd.DataFrame({"TxID": ["T3", "T4"], "Amt": [310, 450]})

# Concatenate vertically, resetting index sequences
vert_concat = pd.concat([df_week1, df_week2], ignore_index=True)
print("--- Vertical Concatenation ---")
print(vert_concat)

# Concatenate horizontally (aligning by row index)
horiz_concat = pd.concat([df_week1, df_week2], axis=1)
print("\n--- Horizontal Concatenation ---")
print(horiz_concat)
```

### Output

```text
--- Vertical Concatenation ---
  TxID  Amt
0   T1  150
1   T2  200
2   T3  310
3   T4  450

--- Horizontal Concatenation ---
  TxID  Amt TxID  Amt
0   T1  150   T3  310
1   T2  200   T4  450
```

---

## Common Mistakes Students Make

- **Unintentional Cartesian products (Many-to-Many joins)**: If a merge key contains duplicate values in both DataFrames, a standard join creates a Cartesian product of those rows. This can inflate the size of the output DataFrame. Always verify key uniqueness using `s.is_unique` before merging.
- **Index duplication in concatenations**: Failing to specify `ignore_index=True` during vertical concatenations preserves duplicate index sequences (e.g. `0, 1, 0, 1`), which can cause indexing issues downstream.
- **Data loss due to default Inner Joins**: Students often write `pd.merge(df_a, df_b)` expecting a complete outer join, forgetting that it defaults to an inner join. This discards rows that don't exist in both tables.
- **Column name conflicts during `.join()`**: Running `df_a.join(df_b)` throws a `ValueError: columns overlap but no suffix specified` if both DataFrames contain columns with the same name. Specify `lsuffix` and `rsuffix` parameters.

---

## Best Practices

- Always use the `how` parameter explicitly inside `pd.merge()` to make the join logic clear.
- Use `ignore_index=True` during vertical concatenations unless the row indices carry specific alignment meaning.
- Set `indicator=True` to check the success rate of left or outer joins during data auditing.
- Verify that keys are unique in at least one DataFrame to prevent memory issues from many-to-many joins.

---

## Worked Real-World Examples

### Worked Example 1: Auditing Loyalty Member Transactions

```python
import pandas as pd

# Loyalty members roster
members = pd.DataFrame({
    "MemberID": ["M_10", "M_20", "M_30"],
    "Tier": ["Gold", "Silver", "Platinum"]
})

# Ingested transactions logs
sales_log = pd.DataFrame({
    "TxID": ["T101", "T102", "T103"],
    "MemberID": ["M_10", "M_20", "M_99"],
    "Spend": [1500, 4200, 800]
})

# Merge to find member details, auditing with indicator flag
audit_sales = pd.merge(members, sales_log, on="MemberID", how="outer", indicator=True)

print("--- Transaction Audit Ledger ---")
print(audit_sales)
```

### Output

```text
--- Transaction Audit Ledger ---
  MemberID      Tier  TxID   Spend      _merge
0     M_10      Gold  T101  1500.0        both
1     M_20    Silver  T102  4200.0        both
2     M_30  Platinum   NaN     NaN   left_only
3     M_99       NaN  T103   800.0  right_only
```
*Note: Member M_99 is flagged as right_only, indicating a transaction from an unregistered member.*

---

### Worked Example 2: Horizontal Segment Alignment

```python
import pandas as pd

# Sensor logs from Node A
sensor_a = pd.DataFrame({"Temp_A": [25.4, 26.1]}, index=["12:00", "13:00"])

# Sensor logs from Node B
sensor_b = pd.DataFrame({"Temp_B": [22.8, 23.4]}, index=["13:00", "14:00"])

# Align columns horizontally by index labels
aligned_telemetry = pd.concat([sensor_a, sensor_b], axis=1)

print("--- Consolidated Telemetry ---")
print(aligned_telemetry)
```

### Output

```text
--- Consolidated Telemetry ---
       Temp_A  Temp_B
12:00    25.4     NaN
13:00    26.1    22.8
14:00     NaN    23.4
```

---

### Worked Example 3: Multi-Table Stack with Segment Keys

```python
import pandas as pd

# Regional transaction registers
sales_east = pd.DataFrame({"TxID": ["E1"], "Amt": [1200]})
sales_west = pd.DataFrame({"TxID": ["W1"], "Amt": [1800]})

# Concatenate vertically, preserving region keys as MultiIndexes
regional_stack = pd.concat([sales_east, sales_west], keys=["East", "West"])

print("--- Hierarchical Regional Stack ---")
print(regional_stack)
```

### Output

```text
--- Hierarchical Regional Stack ---
        TxID   Amt
East 0    E1  1200
West 0    W1  1800
```

---

## Practice Questions

1. Explain the differences between the `pd.merge()` and `.join()` functions in Pandas.
2. Write a command to perform a right outer join between DataFrames `df1` and `df2` using keys `key_1` and `key_2`.
3. How does `how='cross'` behave in `pd.merge()`, and what is its output shape?
4. Write a command to concatenate list of DataFrames `[df1, df2, df3]` horizontally.
5. Explain the purpose and output columns of the `indicator=True` parameter.
6. Write a command to merge two DataFrames on index labels using the `pd.merge()` method.
7. What error occurs when you concatenate DataFrames that share overlapping column names without specifying axis directions?
8. Compare the execution differences of joining using suffixes (`lsuffix`/`rsuffix`) versus `suffixes=('_x', '_y')` parameters.
9. How can you prevent duplicating identical index sequences during vertical concatenations?
10. Describe how to perform a merge on a combination of a column name in the left table and the index in the right table.

---

## Mini Assignments

### Assignment 1: Corporate Personnel Consolidation
- Create an employee roster DataFrame with `EmpID` and `FullName`.
- Create a payroll DataFrame with `StaffID` and `Base_Salary`.
- Create a department DataFrame with `Emp_ID` and `Dept_Name`.
- Merge these tables using a single pipeline, handling mismatched column names and keeping all employee records.

### Assignment 2: Time Series Logs Merging
- Create two irregular telemetry DataFrames containing temperature readings for Sensor A and Sensor B.
- Use `pd.merge()` to combine the logs by index timestamps using an outer join.
- Impute missing values in the combined DataFrame using forward filling.

### Assignment 3: Inventory Consolidation Audit
- Create two regional warehouse registers logging `ItemID`, `Stock_Count`, and `Cost`.
- Stack the tables vertically, creating a MultiIndex to track the source region (e.g. `"East"` or `"West"`).

---

## Interview-Oriented Questions

- **How does `pd.merge()` handle cases where the merge key contains duplicate values in both DataFrames (Many-to-Many joins)?**
  - *Answer*: If the merge key contains duplicates in both DataFrames, a many-to-many join is performed. Pandas computes the Cartesian product of the matching rows. For example, if a key appears $M$ times in the left table and $N$ times in the right table, the output DataFrame will contain $M \times N$ rows for that key, which can cause memory issues if not anticipated.
- **Explain the structural differences between `pd.concat(..., axis=0)` and `pd.concat(..., axis=1)`.**
  - *Answer*: `pd.concat(..., axis=0)` stacks DataFrames vertically along the row axis, aligning them by column names. `pd.concat(..., axis=1)` stacks DataFrames horizontally along the column axis, aligning them by row index labels.
- **What is the difference between `df_left.join(df_right)` and `pd.merge(df_left, df_right, left_index=True, right_index=True)`?**
  - *Answer*: Under the hood, `.join()` calls `pd.merge()` using index alignment. The main difference is syntax: `.join()` is a convenience method that defaults to a left join, whereas `pd.merge()` requires explicit parameters (`left_index=True`, `right_index=True`) and defaults to an inner join.
- **How can we identify which rows in a merged DataFrame came from which source table?**
  - *Answer*: Set `indicator=True` in `pd.merge()`. This adds a column named `_merge` containing values `'left_only'`, `'right_only'`, or `'both'`, indicating the source of each row.
- **What happens when you concatenate two DataFrames vertically (`axis=0`) that have different column names?**
  - *Answer*: The resulting DataFrame will contain the union of the column names. Columns that exist in one DataFrame but not the other are filled with `NaN` for the rows corresponding to the DataFrame that lacks those columns.

---

## Teaching Notes for This Chapter

- **Deconstruct Joins Visually**: Draw Venn diagrams to illustrate inner, outer, left, and right joins.
- **Showcase Cartesian Product Inflations**: Walk students through a many-to-many join example to show how row counts can grow rapidly.
- **Highlight merge indicator utility**: Emphasize how `indicator=True` helps debug failed merges during ETL processes.

---

## Chapter Wrap-up Concepts Students Must Master

- Use `pd.merge()` for database-style joins, connecting rows based on key columns.
- Use `.join()` to combine DataFrames by index labels.
- Use `pd.concat()` to stack datasets vertically (`axis=0`) or horizontally (`axis=1`).
- Set `ignore_index=True` during concatenation to reset the index.
- Use `suffixes` to handle conflicting column names and `indicator=True` to audit join success.
- Many-to-many joins compute the Cartesian product, which can dramatically increase row counts.
