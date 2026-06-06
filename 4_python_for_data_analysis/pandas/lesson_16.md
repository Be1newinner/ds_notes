# Pivot Tables and Crosstabulation

## Lesson Overview

- This chapter covers pivot tables and cross-tabulation in Pandas.
- Pivot tables and cross-tabulations reshape data, shifting columns and rows to summarize distributions. They convert detailed, flat event logs into structured summaries, which are ideal for dashboards and report sheets.
- We will cover `pd.pivot_table()` for aggregating multi-dimensional variables, analyze `pd.crosstab()` for frequency distributions, and implement parameters like `margins=True` and `normalize` to calculate total margins and percentages.
- Mastering these tools allows you to generate business intelligence views and analyze categorical relationships.

## Learning Objectives

- Summarize datasets across multiple dimensions using `pd.pivot_table()`.
- Customize pivot calculations using `aggfunc`, `fill_value`, and row/column summary margins (`margins=True`).
- Analyze frequency distributions between categorical variables using `pd.crosstab()`.
- Normalize cross-tabulations to calculate row, column, or total percentage distributions.
- Distinguish between `.pivot()`, `pd.pivot_table()`, and `pd.crosstab()`.

---

## Pivot Tables with `pd.pivot_table()`

`pd.pivot_table()` aggregates data and reshapes it into a 2D matrix:
- **`index`**: Column(s) to group by along the rows.
- **`columns`**: Column(s) to group by along the columns.
- **`values`**: Column(s) to aggregate.
- **`aggfunc`**: Function(s) used for aggregation (defaults to `'mean'`).
- **`fill_value`**: Value to replace missing cells.
- **`margins`**: Add row and column totals (defaults to `False`).

### Setup for Demonstration

```python
import pandas as pd

# Retail sales transactions
df_retail = pd.DataFrame({
    "Year": [2025, 2025, 2025, 2026, 2026, 2026],
    "Region": ["East", "West", "East", "West", "East", "West"],
    "Product": ["Tech", "Tech", "Home", "Home", "Tech", "Tech"],
    "Sales": [12000, 15000, 8000, 7500, 14000, 18000],
    "Profit": [2000, 3000, 1200, 800, 2500, 4000]
})

print("--- Master Retail Table ---")
print(df_retail)
```

### Output

```text
--- Master Retail Table ---
   Year Region Product  Sales  Profit
0  2025   East    Tech  12000    2000
1  2025   West    Tech  15000    3000
2  2025   East    Home   8000    1200
3  2026   West    Home   7500     800
4  2026   East    Tech  14000    2500
5  2026   West    Tech  18000    4000
```

---

### 1. Basic Pivot Table (Mean Sales by Region and Product)

```python
# Pivot to find average sales
pivot_sales = pd.pivot_table(
    df_retail,
    index="Region",
    columns="Product",
    values="Sales",
    aggfunc="mean"
)

print("--- Average Sales Matrix ---")
print(pivot_sales)
```

### Output

```text
--- Average Sales Matrix ---
Product   Home     Tech
Region                 
East    8000.0  13000.0
West    7500.0  16500.0
```

---

### 2. Multi-Aggregator Pivot Table with Row/Column Margins

Calculate both total sales and total profits, including row/column totals. Replace any missing values with 0.

```python
# Complex pivot table with margins
pivot_complex = pd.pivot_table(
    df_retail,
    index="Region",
    columns="Product",
    values=["Sales", "Profit"],
    aggfunc="sum",
    fill_value=0,
    margins=True
)

print("--- Consolidated Totals with Margins ---")
print(pivot_complex)
```

### Output

```text
--- Consolidated Totals with Margins ---
         Profit             Sales             
Product    Home  Tech   All  Home   Tech   All
Region                                        
East       1200  4500  5700  8000  26000 34000
West        800  7000  7800  7500  33000 40500
All        2000 11500 13500 15500  59000 74500
```
*Note: The `All` column/row displays the consolidated sums across the axes.*

---

## Cross-Tabulation with `pd.crosstab()`

`pd.crosstab()` calculates frequency tables for categorical variables. By default, it counts the occurrences of each combination of categories.

### Setup for Demonstration

```python
# User demographics survey
df_survey = pd.DataFrame({
    "Gender": ["M", "F", "F", "M", "F", "M", "F", "M"],
    "Preference": ["Card", "UPI", "UPI", "Card", "Card", "UPI", "UPI", "Card"]
})

print("--- Master Survey Table ---")
print(df_survey)
```

### Output

```text
--- Master Survey Table ---
  Gender Preference
0      M       Card
1      F        UPI
2      F        UPI
3      M       Card
4      F       Card
5      M        UPI
6      F        UPI
7      M       Card
```

---

### 1. Basic Frequency Count

```python
# Count combinations of Gender and Preference
freq_table = pd.crosstab(df_survey["Gender"], df_survey["Preference"])

print("--- Preference Frequencies ---")
print(freq_table)
```

### Output

```text
--- Preference Frequencies ---
Preference  Card  UPI
Gender               
F              1    3
M              3    1
```

---

### 2. Normalizing Crosstabs

Normalize the counts to display proportions instead of absolute counts:
- `normalize='index'`: Normalize along the rows (each row sums to 1.0 / 100%).
- `normalize='columns'`: Normalize along the columns (each column sums to 1.0).
- `normalize='all'` or `True`: Normalize over all values (the entire table sums to 1.0).

```python
# Normalize by row index to show gender preference ratios
gender_ratios = pd.crosstab(
    df_survey["Gender"],
    df_survey["Preference"],
    normalize="index"
) * 100

print("--- Gender Preference Ratios (%) ---")
print(gender_ratios)
```

### Output

```text
--- Gender Preference Ratios (%) ---
Preference  Card   UPI
Gender                
F           25.0  75.0
M           75.0  25.0
```

---

## `.pivot()` vs `pd.pivot_table()` vs `pd.crosstab()`

- **`.pivot()`**: Reshapes data based on column values **without aggregation**. It raises a `ValueError` if the index/column combinations contain duplicate entries.
- **`pd.pivot_table()`**: Reshapes data and **supports aggregation** (like sum, mean). It is the preferred method when the dataset contains duplicate index/column pairs.
- **`pd.crosstab()`**: Specialized for **categorical frequency counts** and proportions. It accepts array-like inputs directly (e.g. lists, NumPy arrays, or Series) without requiring a pre-existing DataFrame.

---

## Common Mistakes Students Make

- **Using `.pivot()` with duplicate entries**: Attempting to run `df.pivot()` on a dataset that contains duplicate entries for the new index/column layout raises a `ValueError: Index contains duplicate entries, cannot reshape`. Use `pd.pivot_table()` instead.
- **Misinterpreting the default `aggfunc`**: Students often forget that `pd.pivot_table()` defaults to `'mean'`. If they expect total sums, they must specify `aggfunc='sum'` explicitly.
- **Passing a DataFrame column name to `pd.crosstab()` incorrectly**: Running `pd.crosstab('Gender', 'Preference')` evaluates the literal strings instead of the DataFrame columns. You must pass the Series directly: `pd.crosstab(df['Gender'], df['Preference'])`.

---

## Best Practices

- Explicitly specify the `aggfunc` parameter in `pd.pivot_table()` to avoid calculation errors.
- Use `fill_value=0` to replace `NaN`s in the reshaped table, ensuring clean matrices for plotting.
- Use `margins=True` to calculate totals when preparing tables for stakeholders.
- Use `pd.crosstab(..., normalize='index')` to convert raw category counts into proportions.

---

## Worked Real-World Examples

### Worked Example 1: Multi-Store Sales Summary

```python
import pandas as pd

# Product sales logs
sales_log = pd.DataFrame({
    "Store": ["Delhi_1", "Mumbai_1", "Delhi_1", "Delhi_1", "Mumbai_1"],
    "Category": ["Electronics", "Office", "Electronics", "Office", "Electronics"],
    "Revenue": [1200, 450, 1500, 300, 2200]
})

# Generate a pivot table showing total revenue and transaction counts
pivot_summary = pd.pivot_table(
    sales_log,
    index="Store",
    columns="Category",
    values="Revenue",
    aggfunc=["sum", "count"],
    fill_value=0
)

print("--- Store Sales Summary ---")
print(pivot_summary)
```

### Output

```text
--- Store Sales Summary ---
                sum                     count            
Category Electronics Office Electronics Office
Store                                            
Delhi_1         2700    300           2      1
Mumbai_1        2200    450           1      1
```

---

### Worked Example 2: Client Subscription Audit

```python
import pandas as pd

# Active subscriptions
subscriptions = pd.DataFrame({
    "Plan": ["Basic", "Premium", "Basic", "Basic", "Premium", "Standard"],
    "Status": ["Active", "Active", "Cancelled", "Active", "Cancelled", "Active"],
    "Country": ["IN", "US", "IN", "US", "IN", "US"]
})

# Cross-tabulate subscription status by plan type
status_counts = pd.crosstab(
    subscriptions["Plan"],
    subscriptions["Status"],
    margins=True
)

print("--- Subscription Status Distribution ---")
print(status_counts)
```

### Output

```text
--- Subscription Status Distribution ---
Status     Active  Cancelled  All
Plan                             
Basic           2          1    3
Premium         1          1    2
Standard        1          0    1
All             4          2    6
```

---

### Worked Example 3: Normalizing Category Distributions

```python
import pandas as pd

# Device logs
device_logs = pd.DataFrame({
    "Device": ["Mobile", "Desktop", "Mobile", "Mobile", "Desktop"],
    "OS": ["Android", "Windows", "iOS", "Android", "macOS"]
})

# Calculate OS distribution ratios by device type
device_os_dist = pd.crosstab(
    device_logs["Device"],
    device_logs["OS"],
    normalize="index"
) * 100

print("--- OS Proportions by Device (%) ---")
print(device_os_dist)
```

### Output

```text
--- OS Proportions by Device (%) ---
OS       Android       Windows       iOS     macOS
Device                                            
Desktop      0.0  50.0000000     0.0  50.00000
Mobile      66.7         0.0    33.3       0.0
```

---

## Practice Questions

1. Explain the differences between the `.pivot()` and `pd.pivot_table()` functions in Pandas.
2. Write a command to generate a pivot table showing the maximum `Salary` grouped by `Department` (rows) and `Location` (columns).
3. What is the default aggregation function applied by `pd.pivot_table()`?
4. Write a crosstab command to calculate frequency percentages over the entire dataset (`normalize='all'`).
5. Explain the behavior of the `margins` parameter in `pd.pivot_table()`.
6. Write a command to replace all NaN values with `0.0` inside a pivot table.
7. How does `pd.crosstab()` differ from `.groupby()` when analyzing two categorical variables?
8. Write a script that pivots a DataFrame where the columns index is MultiIndexed.
9. Explain how you would pass multiple aggregation functions (e.g. sum and mean) to a single pivot table.
10. Describe how to filter out columns from a pivoted DataFrame that contain more than 50% missing values.

---

## Mini Assignments

### Assignment 1: Corporate Training Pivot
- Create a training log DataFrame containing `Employee`, `Department`, `Course_Type`, and `Hours_Completed`.
- Pivot the dataset to show the total training hours completed by each department for each course type.
- Include margins and fill any missing values with `0`.

### Assignment 2: Customer Payment Survey Crosstab
- Create a survey dataset of 10 users tracking `Age_Group` (`"Under 30"`, `"30 and Over"`) and `Payment_Method` (`"Card"`, `"Cash"`, `"UPI"`).
- Generate a crosstabulation showing payment preference percentages for each age group.

### Assignment 3: Regional Stock Inventory Pivot
- Create a warehouse inventory log tracking: `Warehouse_ID`, `Product_Category`, `Units`, and `Valuation`.
- Generate a pivot table showing both the total units and the average valuation for each category across warehouses.

---

## Interview-Oriented Questions

- **Why does the `.pivot()` method throw an error when the dataset contains duplicate index-column combinations?**
  - *Answer*: `.pivot()` is a reshaping operation that does not aggregate data. If there are duplicate index-column pairs, Pandas cannot determine which value to place in the corresponding cell, raising a `ValueError`. To handle duplicates, use `pd.pivot_table()` and specify an aggregation function.
- **Explain the difference between `normalize='index'` and `normalize='columns'` inside `pd.crosstab()`.**
  - *Answer*: `normalize='index'` divides each cell count by the row total, showing the percentage distribution of categories within each row. `normalize='columns'` divides each cell count by the column total, showing the percentage distribution of categories within each column.
- **How can we aggregate multiple value columns using different functions in a pivot table?**
  - *Answer*: Pass a dictionary to the `aggfunc` parameter. The keys of the dictionary should be the columns to aggregate, and the values should be the aggregation functions. For example: `pd.pivot_table(df, values=['Sales', 'Profit'], aggfunc={'Sales': 'sum', 'Profit': 'mean'})`.
- **Under what scenarios is `pd.crosstab()` preferred over `pd.pivot_table()`?**
  - *Answer*: `pd.crosstab()` is preferred when working directly with raw categorical arrays or Series without first converting them into a DataFrame. It is also optimized for counting frequencies and calculating percentages across categorical columns.
- **Can we pivot a DataFrame along multiple index columns?**
  - *Answer*: Yes, you can pass a list of column names to the `index` parameter of `pd.pivot_table()`. This creates a MultiIndexed row layout in the output table.

---

## Teaching Notes for This Chapter

- **Deconstruct Reshaping Geometry**: Draw a long dataset on the board and trace how row values are redistributed to form the columns of a wider table.
- **Contrast pivot variants**: Clearly illustrate when to use `.pivot()`, `pd.pivot_table()`, and `pd.crosstab()` using a comparison table.
- **Explain the role of margins**: Show how `margins=True` calculates row and column totals, helping students verify calculations.

---

## Chapter Wrap-up Concepts Students Must Master

- Use `pd.pivot_table()` to reshape and aggregate data, and use `fill_value` to handle missing entries.
- `pd.pivot_table()` defaults to `'mean'`; specify `aggfunc='sum'` explicitly when calculating totals.
- Use `pd.crosstab()` to calculate frequency distributions for categorical columns.
- Normalize cross-tabulations using `normalize='index'` or `normalize='columns'` to show percentages.
- Set `margins=True` to include total rows and columns in summary tables.
