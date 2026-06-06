# Arithmetic Operations and Alignment on Series and DataFrames

## Lesson Overview

- This chapter explores index alignment and broadcasting during arithmetic operations in Pandas.
- Unlike traditional arrays (like NumPy), Pandas automatically aligns index labels of two datasets before applying mathematical operations (addition, subtraction, etc.). If labels do not match, the resulting union of indexes is filled with `NaN` values.
- We will cover standard operators (`+`, `-`, `*`, `/`), explore arithmetic methods (`.add()`, `.sub()`, `.mul()`, `.div()`) that accept a `fill_value` parameter to prevent data loss, and analyze broadcasting rules between 2D DataFrames and 1D Series.
- Mastering alignment rules ensures that financial ledger adjustments, scientific scaling, and data transformations remain accurate even when working with mismatched tables.

## Learning Objectives

- Understand index alignment and how Pandas performs mathematical operations on the union of index labels.
- Prevent missing entries (`NaN`) in arithmetic outcomes using flexible arithmetic methods (`.add()`, `.sub()`, `.mul()`, `.div()`) with a `fill_value`.
- Implement broadcasting rules to apply arithmetic operations between a 2D DataFrame and a 1D Series.
- Align operations vertically across rows (`axis='index'`) or horizontally across columns (`axis='columns'`).
- Avoid common mathematical pitfalls during alignment mismatches.

---

## The Geometry of Index Alignment

When you perform arithmetic operations on two Pandas objects (Series or DataFrames), Pandas aligns the objects by their index labels. The result contains the union of the index labels from both objects.
- **Matching Labels**: The mathematical operation is applied to the values.
- **Mismatched Labels**: The operation evaluates to `NaN` because the value is missing in one of the objects.

### Setup for Demonstration

```python
import pandas as pd

# Revenue series for two different months
rev_may = pd.Series([100, 200, 300], index=["Store_A", "Store_B", "Store_C"])
rev_june = pd.Series([150, 250, 350], index=["Store_B", "Store_C", "Store_D"])

print("--- May Revenue ---")
print(rev_may)
print("\n--- June Revenue ---")
print(rev_june)
```

### Output

```text
--- May Revenue ---
Store_A    100
Store_B    200
Store_C    300
dtype: int64

--- June Revenue ---
Store_B    150
Store_C    250
Store_D    350
dtype: int64
```

---

### 1. Basic Addition Operator (`+`)

```python
# Add series using standard operator
total_rev = rev_may + rev_june

print("--- Total Revenue (Operator) ---")
print(total_rev)
```

### Output

```text
--- Total Revenue (Operator) ---
Store_A      NaN
Store_B    350.0
Store_C    550.0
Store_D      NaN
dtype: float64
```
*Notice that Store_A and Store_D have evaluated to `NaN` because their labels were not present in both Series.*

---

## 2. Using Arithmetic Methods with `fill_value`

To prevent mismatched labels from evaluating to `NaN`, use Pandas' flexible arithmetic methods instead of standard operators:
- `.add()`: Addition (`+`)
- `.sub()`: Subtraction (`-`)
- `.mul()`: Multiplication (`*`)
- `.div()`: Division (`/`)

These methods accept a `fill_value` parameter. If a label is missing in one of the objects, Pandas replaces the missing value with the `fill_value` before performing the operation.

```python
# Add series using .add() with fill_value=0
total_rev_filled = rev_may.add(rev_june, fill_value=0)

print("--- Total Revenue (add with fill_value=0) ---")
print(total_rev_filled)
```

### Output

```text
--- Total Revenue (add with fill_value=0) ---
Store_A    100.0
Store_B    350.0
Store_C    550.0
Store_D    350.0
dtype: float64
```
*Note: If a label is missing in BOTH objects, the result remains `NaN`.*

---

## 3. Operations Between DataFrames and Series (Broadcasting)

Operations between a 2D DataFrame and a 1D Series align by index or columns and broadcast the Series values across the matching axis.

### Default Broadcasting Behavior (axis='columns')
By default, operations between a DataFrame and a Series align the Series index with the DataFrame **columns** and broadcast the operation down the rows.

```python
# Sample sales DataFrame
df_sales = pd.DataFrame({
    "Electronics": [1200, 800, 1500],
    "Apparel": [450, 310, 600]
}, index=["Day_1", "Day_2", "Day_3"])

# Tax rates Series (matching column names)
tax_rates = pd.Series([1.18, 1.05], index=["Electronics", "Apparel"])

print("--- Sales DataFrame ---")
print(df_sales)

print("\n--- Broadcasted Tax Addition ---")
print(df_sales.mul(tax_rates, axis="columns"))
```

### Output

```text
--- Sales DataFrame ---
       Electronics  Apparel
Day_1         1200      450
Day_2          800      310
Day_3         1500      600

--- Broadcasted Tax Addition ---
       Electronics  Apparel
Day_1       1416.0    472.5
Day_2        944.0    325.5
Day_3       1770.0    630.0
```

---

### Vertical Broadcasting (axis='index')
To align the Series index with the DataFrame **row index** and broadcast the operation across the columns, specify `axis='index'` or `axis=0`.

```python
# Daily store discounts Series (matching row index)
daily_discounts = pd.Series([50, 20, 100], index=["Day_1", "Day_2", "Day_3"])

# Apply discounts down the rows
discounted_sales = df_sales.sub(daily_discounts, axis="index")

print("--- Vertical Broadcasted Discounts ---")
print(discounted_sales)
```

### Output

```text
--- Vertical Broadcasted Discounts ---
       Electronics  Apparel
Day_1         1150      400
Day_2          780      290
Day_3         1400      500
```

---

## Common Mistakes Students Make

- **Using standard operators and getting unexpected NaNs**: Students often write `df_a + df_b` and are surprised when the output is mostly `NaN` because the row or column indexes were not perfectly aligned. Always use `.add(..., fill_value=0)` when working with datasets that may have mismatched keys.
- **Confusing axis directions in broadcasting**: Running `df.sub(series)` without specifying the axis will raise a `ValueError` or return all `NaN`s if the Series index matches the DataFrame rows instead of its columns. Remember that the default broadcasting axis is `axis='columns'` (or `axis=1`).
- **Dividing by Zero**: Using `/` or `.div()` where the divisor series has a `0` value results in `inf` (Infinity) or `-inf`, which can corrupt statistical aggregates downstream. Clean zero values before dividing or use `.replace(0, np.nan)`.

---

## Best Practices

- Standardize on `.add()`, `.sub()`, `.mul()`, and `.div()` instead of operator symbols (`+`, `-`, `*`, `/`) when working with mismatched tables to explicitly handle missing entries using `fill_value`.
- Always verify axis parameters when broadcasting operations between DataFrames and Series.
- When performing subtraction or division, check if the data contains zeros or negatives to prevent division errors.

---

## Worked Real-World Examples

### Worked Example 1: Consolidated Sales Inventory Audit

```python
import pandas as pd

# Inventory at Warehouse A
inv_a = pd.DataFrame({
    "Product_X": [15, 20],
    "Product_Y": [5, 10]
}, index=["Shelf_1", "Shelf_2"])

# Inventory at Warehouse B (mismatched shelf layout)
inv_b = pd.DataFrame({
    "Product_X": [10, 5],
    "Product_Y": [8, 12]
}, index=["Shelf_2", "Shelf_3"])

# Sum inventories, filling missing shelf counts with 0
total_inventory = inv_a.add(inv_b, fill_value=0)

print("--- Total Consolidated Stock ---")
print(total_inventory)
```

### Output

```text
--- Total Consolidated Stock ---
         Product_X  Product_Y
Shelf_1       15.0        5.0
Shelf_2       30.0       18.0
Shelf_3        5.0       12.0
```

---

### Worked Example 2: Normalizing Sales Matrix by Region Target

```python
import pandas as pd

# Monthly regional sales revenue
sales_grid = pd.DataFrame({
    "Q1": [12000, 15000, 8000],
    "Q2": [14000, 13000, 9500]
}, index=["Region_East", "Region_West", "Region_South"])

# Target budget metrics per region
regional_targets = pd.Series([10000, 12000, 9000], index=["Region_East", "Region_West", "Region_South"])

# Calculate achievement percentage (Sales / Target)
achievement_pct = sales_grid.div(regional_targets, axis="index") * 100

print("--- Regional Target Achievement (%) ---")
print(achievement_pct)
```

### Output

```text
--- Regional Target Achievement (%) ---
                   Q1          Q2
Region_East   120.000  140.000000
Region_West   125.000  108.333333
Region_South   88.889  105.555556
```

---

### Worked Example 3: Scaling Pricing Matrix with Discount Factor

```python
import pandas as pd

# In-store pricing sheet
df_prices = pd.DataFrame({
    "Base_Price": [100.0, 250.0, 50.0],
    "Shipping": [15.0, 25.0, 5.0]
})

# Flat discount rate multiplier
discount_factor = 0.90  # 10% off

# Apply discount factor across all columns
discounted_prices = df_prices.mul(discount_factor)

print("--- Discounted Catalog ---")
print(discounted_prices)
```

### Output

```text
--- Discounted Catalog ---
   Base_Price  Shipping
0        90.0      13.5
1       225.0      22.5
2        45.0       4.5
```

---

## Practice Questions

1. Explain how index alignment handles mismatched keys during addition.
2. Write a command to subtract Series `s1` from Series `s2`, filling missing labels with `0` before operating.
3. Define broadcasting and explain how Pandas handles it between a 2D DataFrame and a 1D Series.
4. Write a script that multiplies DataFrame `df` by a Series `s` aligned along the row index.
5. What does the expression `df_a.div(df_b, fill_value=1)` return for index labels that are missing in `df_a` but present in `df_b`?
6. Write a command to divide all elements in a DataFrame by 100.
7. Explain the default axis setting when broadcasting arithmetic operations in Pandas.
8. How does Pandas handle a division by zero operation in elements?
9. Compare the result of `df_a + df_b` where `df_a` has duplicate index labels and `df_b` has unique index labels.
10. Describe how to perform element-wise modulo operations between two DataFrames.

---

## Mini Assignments

### Assignment 1: Sales Tax and Shipping Consolidation
- Create two regional sales DataFrames: `df_east` and `df_west`, tracking sales for three product types. Mismatch the product indexes slightly.
- Sum the two regional sales, filling missing product values with `0`.
- Broadcast a tax deduction Series (`axis='columns'`) to calculate the net-of-tax sales amounts.

### Assignment 2: Time Series Temperature Calibration
- Create an irregular Series tracking temperature readings at specific timestamps for Sensor A and Sensor B.
- Subtract Sensor B readings from Sensor A readings to calculate the differences, filling missing timestamps with a baseline temperature of `20.0` degrees.

### Assignment 3: Inventory Valuation Matrix
- Create an inventory quantity DataFrame for 4 products across 3 shelves.
- Create a price Series mapping unique unit costs for each product.
- Multiply the quantities by the prices using broadcasting to output the total inventory valuation matrix.

---

## Interview-Oriented Questions

- **What does index alignment mean in Pandas, and what are its core advantages over standard NumPy arrays?**
  - *Answer*: Index alignment means that operations between Pandas objects are aligned automatically using the index labels rather than the element positions. The main advantage is that it prevents data misalignment errors when merging or operating on datasets of different shapes or sorted states, as Pandas automatically matches the labels and returns the union of the keys.
- **Explain the difference between `df.sub(series, axis='index')` and `df.sub(series, axis='columns')`.**
  - *Answer*: `df.sub(series, axis='index')` aligns the Series index with the DataFrame row index and subtracts the Series values down each column. `df.sub(series, axis='columns')` aligns the Series index with the DataFrame column names and subtracts the values across each row.
- **How does the `fill_value` parameter behave when both DataFrames lack a specific index label?**
  - *Answer*: If a label is missing in both DataFrames, the output at that position remains `NaN`. The `fill_value` is only used if the label exists in at least one of the objects being aligned.
- **What is the mathematical outcome of applying operations like `+` or `-` on Series with duplicate index labels?**
  - *Answer*: If there are duplicate labels, Pandas performs a Cartesian product of the matching labels. For example, if a label appears $M$ times in the first Series and $N$ times in the second, the output will contain $M \times N$ rows for that label.
- **How does Pandas handle operations when data types differ (e.g. adding an integer Series to a float Series)?**
  - *Answer*: Pandas automatically promotes the data types to prevent precision loss. In this case, the integer Series values are cast to floats before the addition, and the resulting Series has a `float64` data type.

---

## Teaching Notes for This Chapter

- **Deconstruct Cartesian Product Alignment**: Spend 5 minutes illustrating what happens when adding Series with duplicate index labels to prevent student confusion.
- **Visualize Broadcasting**: Use a diagram showing how a 1D Series is stretched (broadcasted) across a 2D DataFrame grid to match its shape.
- **Emphasize method names**: Remind students that methods like `.add()`, `.sub()`, `.mul()`, and `.div()` are safer than operators when data alignment is imperfect.

---

## Chapter Wrap-up Concepts Students Must Master

- Pandas performs arithmetic operations on the union of index labels, returning `NaN` for mismatched labels.
- Use arithmetic methods (`.add()`, `.sub()`, `.mul()`, `.div()`) with `fill_value` to handle mismatched labels safely.
- Operations between DataFrames and Series broadcast values, defaulting to aligning columns (`axis='columns'`).
- Specify `axis='index'` to broadcast operations vertically down the rows.
- Duplicate labels align by performing a Cartesian product, increasing the number of output rows.
