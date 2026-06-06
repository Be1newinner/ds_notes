# Sorting, Ranking, and Reindexing Data

## Lesson Overview

- This chapter focuses on re-organizing and aligning data in Pandas. We explore sorting indexes and values, ranking array elements, and conforming existing indexes to new target schemes.
- Data presentation and alignment are vital steps in exploratory data analysis. Sorting lets you quickly identify extremes, ranking orders data without altering its layout, and reindexing aligns multiple datasets to a unified index.
- Confusing the behavior of `.sort_values()` and `.sort_index()`, misunderstanding `.rank()` tie-breaking methods, or experiencing data loss during misaligned `.reindex()` operations are frequent challenges for students.
- Mastering these concepts allows you to cleanly prep datasets for time series alignment, modeling, and dashboard presentations.

## Learning Objectives

- Sort Series and DataFrames by index labels or column values in ascending or descending order.
- Customize sorting behaviors using multi-column criteria and define custom NaN placement (`first` or `last`).
- Apply the `.rank()` function and evaluate the performance of different tie-breaking strategies (average, min, max, first, dense).
- Restructure DataFrames using `.reindex()` to align row/column axes to external schemas.
- Implement interpolation methods (forward fill, backward fill, constant fills) during reindexing to handle missing entries.
- Diagnose and resolve common alignment errors during sorting and reindexing.

---

## Sorting in Pandas

Pandas supports sorting along both axes: sorting by index labels (`.sort_index()`) and sorting by actual data values (`.sort_values()`).

### Setup for Demonstration

```python
import pandas as pd
import numpy as np

# Sample dataset representing employee performance metrics
df_emp = pd.DataFrame({
    "Name": ["Vijay", "Anjali", "Siddharth", "Pooja", "Vikram"],
    "Department": ["Sales", "Tech", "Sales", "HR", "Tech"],
    "Salary": [65000, 95000, 65000, np.nan, 88000],
    "Experience": [3, 5, 2, 1, 4]
}, index=["E_103", "E_101", "E_105", "E_104", "E_102"])

print("--- Master Employee Table ---")
print(df_emp)
```

### Output

```text
--- Master Employee Table ---
         Name Department   Salary  Experience
E_103     Vijay      Sales  65000.0           3
E_101    Anjali       Tech  95000.0           5
E_105 Siddharth      Sales  65000.0           2
E_104     Pooja         HR      NaN           1
E_102    Vikram       Tech  88000.0           4
```

---

### 1. Sorting by Index (`.sort_index`)

Sort rows or columns by label names alphabetically or numerically.

```python
# Sort rows by index labels ascending
sorted_by_idx = df_emp.sort_index()
print("--- Sorted by Index (Row labels) ---")
print(sorted_by_idx)
```

### Output

```text
--- Sorted by Index (Row labels) ---
         Name Department   Salary  Experience
E_101    Anjali       Tech  95000.0           5
E_102    Vikram       Tech  88000.0           4
E_103     Vijay      Sales  65000.0           3
E_104     Pooja         HR      NaN           1
E_105 Siddharth      Sales  65000.0           2
```

---

### 2. Sorting by Values (`.sort_values`)

Sort records by column values. When sorting multiple columns, specify lists for the `by` and `ascending` parameters.

```python
# Sort by Salary descending, and place missing values at the end (default)
sorted_salary = df_emp.sort_values(by="Salary", ascending=False, na_position="last")
print("--- Sorted by Salary Descending ---")
print(sorted_salary)

# Sort by Department ascending, and then by Experience descending within departments
multi_sorted = df_emp.sort_values(by=["Department", "Experience"], ascending=[True, False])
print("\n--- Multi-Column Sorted Table ---")
print(multi_sorted)
```

### Output

```text
--- Sorted by Salary Descending ---
         Name Department   Salary  Experience
E_101    Anjali       Tech  95000.0           5
E_102    Vikram       Tech  88000.0           4
E_103     Vijay      Sales  65000.0           3
E_105 Siddharth      Sales  65000.0           2
E_104     Pooja         HR      NaN           1

--- Multi-Column Sorted Table ---
         Name Department   Salary  Experience
E_104     Pooja         HR      NaN           1
E_103     Vijay      Sales  65000.0           3
E_105 Siddharth      Sales  65000.0           2
E_101    Anjali       Tech  95000.0           5
E_102    Vikram       Tech  88000.0           4
```

---

## Ranking Data with `.rank()`

Ranking computes numerical ranks (1 through N) for elements in an array. Tied values are assigned ranks based on a tie-breaking method.

### Tie-Breaking Methods

- `average`: Assign the average rank of the group of tied elements (default).
- `min`: Assign the minimum rank of the group of tied elements (classic sport ranking).
- `max`: Assign the maximum rank of the group.
- `first`: Assign ranks in the order they appear in the dataset.
- `dense`: Like `min`, but rank always increases by 1 between groups instead of skipping ranks.

```python
# Extract salaries
salaries = df_emp[["Name", "Salary"]].copy()

# Add ranks using different methods
salaries["Rank_Average"] = salaries["Salary"].rank(method="average")
salaries["Rank_Min"] = salaries["Salary"].rank(method="min")
salaries["Rank_First"] = salaries["Salary"].rank(method="first")
salaries["Rank_Dense"] = salaries["Salary"].rank(method="dense")

print("--- Comparison of Ranking Methods ---")
print(salaries.sort_values(by="Salary"))
```

### Output

```text
--- Comparison of Ranking Methods ---
         Name   Salary  Rank_Average  Rank_Min  Rank_First  Rank_Dense
E_103     Vijay  65000.0           1.5       1.0         1.0         1.0
E_105 Siddharth  65000.0           1.5       1.0         2.0         1.0
E_102    Vikram  88000.0           3.0       3.0         3.0         2.0
E_101    Anjali  95000.0           4.0       4.0         4.0         3.0
E_104     Pooja      NaN           NaN       NaN         NaN         NaN
```
*Notice that Vijay and Siddharth have identical salaries of 65000. Under `average` they both get 1.5. Under `min` they both get 1.0 (ranking continues at 3.0). Under `dense` they both get 1.0 and ranking continues at 2.0.*

---

## Reindexing with `.reindex()`

Reindexing conforms a Series or DataFrame index to a new target index. If an index label was not present, a null value `NaN` is inserted.

```python
# Define new index sequence (containing some existing keys and some new keys)
new_index_list = ["E_101", "E_102", "E_106", "E_103"]

# Reindex row labels
reindexed_emp = df_emp.reindex(new_index_list)
print("--- Reindexed Employee Table ---")
print(reindexed_emp)
```

### Output

```text
--- Reindexed Employee Table ---
         Name Department   Salary  Experience
E_101  Anjali       Tech  95000.0         5.0
E_102  Vikram       Tech  88000.0         4.0
E_106     NaN        NaN      NaN         NaN
E_103   Vijay      Sales  65000.0         3.0
```

---

### Filling Values during Reindexing

When reindexing ordered data (like timestamps), you can fill in missing entries using interpolation:
- `ffill` / `pad`: Forward fill values from the last valid observation.
- `bfill` / `backfill`: Backward fill values from the next valid observation.
- `fill_value`: Set a constant replacement value for missing targets.

```python
# Simple ordered series
s_prices = pd.Series([10.5, 12.0], index=[1, 3])

print("--- Original Series ---")
print(s_prices)

# Reindex integers 1 through 4
print("\n--- Reindexed with Forward Fill ---")
print(s_prices.reindex(range(1, 5), method="ffill"))

print("\n--- Reindexed with Constant Fill ---")
print(s_prices.reindex(range(1, 5), fill_value=0.0))
```

### Output

```text
--- Original Series ---
1    10.5
3    12.0
dtype: float64

--- Reindexed with Forward Fill ---
1    10.5
2    10.5
3    12.0
4    12.0
dtype: float64

--- Reindexed with Constant Fill ---
1    10.5
2     0.0
3    12.0
4     0.0
dtype: float64
```

---

## Common Mistakes Students Make

- **Using `.reindex` when they mean `.sort_index`**: Attempting to sort a DataFrame using `df.reindex(sorted(df.index))` is slow and manual. Always use `df.sort_index()`.
- **Accidental Data Loss during Reindexing**: Reindexing with labels that do not overlap with the original index will result in a DataFrame populated entirely with `NaN`s. Double-check that the target index labels align with the source labels.
- **Forgetting `inplace=False` by Default**: Writing `df.sort_values("Col")` does not save the sorted state unless re-assigned (`df = df.sort_values(...)`) or using `inplace=True`.
- **Misapplying interpolation on unsorted indexes**: Using `method='ffill'` on an unsorted index during `.reindex()` raises a `ValueError: Image index must be monotonic increasing or decreasing`. Ensure the index is sorted before running interpolation methods.

---

## Best Practices

- Sort datasets immediately before running sequence-dependent analytics (like rolling calculations or forward filling).
- Prefer `.sort_values()` over manual index alignment for structural reorganizations.
- Standardize on explicit column naming when sorting multiple targets: `df.sort_values(by=["A", "B"], ascending=[True, False])`.
- Use `.rank(method='dense')` when you need ordinal ratings that don't skip ranks during duplicates.

---

## Worked Real-World Examples

### Worked Example 1: Leaderboard Generation

```python
import pandas as pd

# Standard e-sports registration
leaderboard = pd.DataFrame({
    "Username": ["Sniper9X", "ProGamer", "ShadowRider", "CyberPunk", "NoobMaster"],
    "Score": [1420, 1550, 1420, 1890, 890]
})

# Sort descending to find rank
leaderboard = leaderboard.sort_values(by="Score", ascending=False).reset_index(drop=True)

# Generate dense positions
leaderboard["Position"] = leaderboard["Score"].rank(method="min", ascending=False).astype(int)

print("--- Final E-Sports Leaderboard ---")
print(leaderboard)
```

### Output

```text
--- Final E-Sports Leaderboard ---
      Username  Score  Position
0    CyberPunk   1890         1
1     ProGamer   1550         2
2     Sniper9X   1420         3
3  ShadowRider   1420         3
4   NoobMaster    890         5
```

---

### Worked Example 2: Financial Time Series Alignment

```python
import pandas as pd

# Irregular stock prices
stock_a = pd.Series([150.20, 151.10], index=pd.to_datetime(["2026-06-01", "2026-06-03"]))

# Reference calendar mapping every day of the week
all_days = pd.date_range(start="2026-06-01", end="2026-06-04", freq="D")

# Conform dataset to complete daily record
stock_a_daily = stock_a.reindex(all_days, method="ffill")
print("--- Aligned Stock Prices ---")
print(stock_a_daily)
```

### Output

```text
--- Aligned Stock Prices ---
2026-06-01    150.20
2026-06-02    150.20
2026-06-03    151.10
2026-06-04    151.10
Freq: D, dtype: float64
```

---

### Worked Example 3: Reindexing Column Labels

```python
import pandas as pd

df_raw = pd.DataFrame({
    "Age": [25, 30],
    "Name": ["Aarav", "Neha"]
})

# Reorder columns and add a missing column with a default value
columns_target = ["Name", "Age", "City"]
df_conformed = df_raw.reindex(columns=columns_target, fill_value="Unknown")

print("--- Conformed Column Matrix ---")
print(df_conformed)
```

### Output

```text
--- Conformed Column Matrix ---
    Name  Age     City
0  Aarav   25  Unknown
1   Neha   30  Unknown
```

---

## Practice Questions

1. Describe the key functional difference between `.sort_values()` and `.sort_index()`.
2. Write a command to sort a DataFrame `df` by a column named `Age` in descending order, putting all missing values (`NaN`) first.
3. How does the `method='dense'` ranking parameter handle duplicate ranks compared to `method='min'`?
4. Write a script showing the error returned when invoking reindexing with `method='ffill'` on an unsorted index Series.
5. Explain how columns can be reordered using `.reindex()` instead of standard bracket access `df[[...]]`.
6. What is the return value of `.rank(ascending=False)` when applied to a Series containing strings?
7. Write a statement to sort a DataFrame index in descending order.
8. How does `fill_value` behave inside `.reindex()` when a conformed index target matches an existing source index label?
9. Compare the execution behavior of `.reindex()` vs `.reset_index()`.
10. Describe how the tie-breaking method `method='first'` assigns ranks to duplicates.

---

## Mini Assignments

### Assignment 1: Academic Grading Rankings
- Create a classroom grade sheet of 10 students tracking: `Student_Name`, `Exam_Score` (out of 100), and `Subject`. Add duplicate scores of `85` and `90` to observe ranking ties.
- Generate ranks for `Exam_Score` using both `min` and `dense` methodologies.
- Sort the gradesheet by `Exam_Score` descending, reset the index, and display the comparison table.

### Assignment 2: E-commerce Product Catalogue Alignment
- Create a product inventory DataFrame containing column fields: `SKU`, `Product_Group`, and `Base_Price`. Use string indexes `P_01` to `P_05`.
- Reindex the DataFrame using target keys `P_01`, `P_03`, `P_06`, `P_02`, filling missing elements with a `Base_Price` of `0.0`.

### Assignment 3: Hourly Telemetry Normalization
- Create a Series tracking hourly temperatures with index timestamps: `10:00`, `11:00`, and `13:00`.
- Normalize the timeline to include `10:00`, `11:00`, `12:00`, and `13:00`.
- Interpolate the missing temperature value for `12:00` using the forward fill method.

---

## Interview-Oriented Questions

- **Explain the exact behavioral differences among standard tie-breaking methods in Pandas `.rank()`.**
  - *Answer*: If two elements are tied for positions 1 and 2: `average` assigns rank 1.5 to both; `min` assigns 1.0 to both (skips to 3); `max` assigns 2.0 to both (skips to 3); `first` assigns 1.0 to the first appearance and 2.0 to the second; `dense` assigns 1.0 to both (next rank is 2.0).
- **Why does attempting to reindex a DataFrame using `.reindex()` with duplicate labels in the source index throw an error?**
  - *Answer*: Reindexing relies on unique index mapping. If the source index contains duplicate keys, Pandas cannot resolve which source element aligns to the target key, raising a `ValueError: cannot reindex on an axis with duplicate labels`.
- **What is the difference between sorting in-place using `inplace=True` and re-assigning the returned output of `.sort_values()`?**
  - *Answer*: `inplace=True` modifies the underlying object memory block directly and returns `None`, which prevents method chaining. Re-assignment creates a copy with the sorted modifications and returns it.
- **How can we utilize `.reindex()` to perform simple forward and backward data imputation across mismatched date metrics?**
  - *Answer*: By passing `method='ffill'` or `method='bfill'` to `.reindex()`, Pandas aligns the index to the target dates and propagates the last or next valid observation to fill missing entries.
- **Explain how the `na_position` parameter in `.sort_values()` impacts analytical summaries.**
  - *Answer*: It dictates where missing values (`NaN`) are placed post-sorting. Setting `na_position='first'` puts them at the top of the sorted output, while `'last'` places them at the bottom, helping developers isolate or ignore nulls during analysis.

---

## Teaching Notes for This Chapter

- **Visualize Ranking Math**: Draw a ladder diagram on the board to illustrate how rank indexes are skipped in `min` vs preserved in `dense`.
- **Trace Reindexing Failure Modes**: Show what happens when a target index has zero overlap with the original index (returns all nulls).
- **Showcase Monotonicity**: Remind students that interpolation methods (`ffill`, `bfill`) require the source index to be sorted (monotonic) or they will fail.

---

## Chapter Wrap-up Concepts Students Must Master

- Use `.sort_index()` to sort row or column headers and `.sort_values()` to sort actual data.
- The `.rank()` method creates numeric placements, resolving ties using `average`, `min`, `max`, `first`, or `dense`.
- Reindexing conforms a DataFrame or Series to a target index label array, inserting `NaN` for missing targets.
- Forward fill (`ffill`) and backward fill (`bfill`) require monotonic indexes to interpolate missing segments.
