# Categorical Data and Efficient Representations

## Lesson Overview

- This chapter covers categorical data types in Pandas. We explore memory optimization, ordering categories dynamically, and accessing properties using the `.cat` accessor.
- Real-world datasets often contain text fields with repeating values (e.g. regions, branch departments, educational grades, or customer tiers). Storing these as standard string objects is inefficient, consuming excessive memory and slowing down sorting and grouping operations.
- We will cover casting columns to `category`, defining custom order rankings using `pd.CategoricalDtype`, and manipulating values using `.cat` methods.
- Mastering categorical types reduces memory footprints and ensures correct sorting order.

## Learning Objectives

- Optimize memory usage in DataFrames by converting string columns with repeating values to the `category` type.
- Define custom sorting sequences for ordinal variables using `pd.CategoricalDtype` with `ordered=True`.
- Access and modify category labels and underlying numeric codes using the `.cat` accessor.
- Perform high-speed sorting and grouping operations using categorical columns.
- Avoid common pitfalls, such as memory bloating from high-cardinality categorical conversions.

---

## The Architecture of Categorical Data

When a column is cast to `category`, Pandas splits the data into two components:
1. **Categories**: A unique array of distinct string values (index lookup table).
2. **Codes**: An array of integers mapping each row to its category index.

```text
Object Representation (Inefficient)          Categorical Representation (Optimized)
Row 0: "Electronics"                          Lookup Table (Categories):
Row 1: "Office"                               Code 0 -> "Electronics"
Row 2: "Electronics"               =====>     Code 1 -> "Office"
Row 3: "Office"                               
Row 4: "Electronics"                          Data (Codes):
                                              [0, 1, 0, 1, 0]
```

### Setup for Demonstration

```python
import pandas as pd

# Customer registry with repeating tiers
df_cust = pd.DataFrame({
    "Name": ["Aarav", "Neha", "Vikram", "Pooja", "Rajesh"],
    "Tier": ["Gold", "Silver", "Gold", "Bronze", "Silver"]
})

print("--- Original Tier Type ---")
print(df_cust["Tier"].dtype)
```

### Output

```text
--- Original Tier Type ---
object
```

---

## 1. Defining Ordered Categorical Types

By default, converting a column to `'category'` using `.astype('category')` assigns categories alphabetically, with no inherent logical order (nominal categories).

For ordinal variables (where order matters, e.g. Bronze < Silver < Gold), define a custom order using `pd.CategoricalDtype`.

```python
# Define the ordered category type
tier_type = pd.CategoricalDtype(
    categories=["Bronze", "Silver", "Gold"],
    ordered=True
)

# Apply the custom type to the column
df_cust["Tier"] = df_cust["Tier"].astype(tier_type)

print("--- Categorical Tier Details ---")
print(df_cust["Tier"])
print("New Type:", df_cust["Tier"].dtype)
```

### Output

```text
--- Categorical Tier Details ---
0      Gold
1    Silver
2      Gold
3    Bronze
4    Silver
Name: Tier, dtype: category
Categories (3, object): ['Bronze' < 'Silver' < 'Gold']
New Type: category
```

---

## 2. Accessing Categorical Properties with the `.cat` Accessor

Vectorized operations on categorical columns require the **`.cat`** accessor:
- `s.cat.categories`: View the unique category labels.
- `s.cat.codes`: View the underlying integer codes mapping to each row.
- `s.cat.add_categories()`: Add new category labels to the lookup table.
- `s.cat.remove_categories()`: Remove category labels.
- `s.cat.rename_categories()`: Rename existing category labels.

```python
# View categories and codes
print("Category Labels:", df_cust["Tier"].cat.categories.tolist())
print("Underlying Codes:", df_cust["Tier"].cat.codes.tolist())

# Add a new category level 'Platinum'
df_cust["Tier"] = df_cust["Tier"].cat.add_categories("Platinum")
print("\n--- After Adding Platinum ---")
print(df_cust["Tier"].cat.categories.tolist())
```

### Output

```text
Category Labels: ['Bronze', 'Silver', 'Gold']
Underlying Codes: [2, 1, 2, 0, 1]

--- After Adding Platinum ---
['Bronze', 'Silver', 'Gold', 'Platinum']
```

---

## 3. Sorting and Grouping Ordered Categorical Data

Because the `Tier` column is ordered (`Bronze < Silver < Gold < Platinum`), sorting operations will follow this custom sequence rather than alphabetical order.

```python
# Sort by Tier (custom sequence)
sorted_cust = df_cust.sort_values(by="Tier")
print("--- Sorted by Tier ---")
print(sorted_cust)

# Group by Tier (retains empty categories by default in older versions, customizable via observed)
grouped_cust = df_cust.groupby("Tier", observed=False).size()
print("\n--- Grouped Tier Counts ---")
print(grouped_cust)
```

### Output

```text
--- Sorted by Tier ---
     Name    Tier
3   Pooja  Bronze
1    Neha  Silver
4  Rajesh  Silver
0   Aarav    Gold
2  Vikram    Gold

--- Grouped Tier Counts ---
Tier
Bronze      1
Silver      2
Gold        2
Platinum    0
dtype: int64
```
*Note: Platinum has a count of 0 but remains in the grouped summary because it is a defined category. Set `observed=True` in `.groupby()` to exclude categories with zero occurrences.*

---

## Common Mistakes Students Make

- **Converting high-cardinality columns**: Converting columns where almost every row is unique (e.g. `Email`, `Invoice_ID`, or `UUID`) to `category` can increase memory usage and slow down operations because Pandas has to store a massive lookup table. Only use categories for columns with a low percentage of unique values.
- **Losing category ordering during operations**: Applying string methods (like `df['Tier'].str.upper()`) converts the column back to a standard `object` type, stripping the categories and custom ordering. Apply modifications using `.cat` methods (like `.cat.rename_categories`) instead.
- **Merging mismatched categorical categories**: Merging two DataFrames on categorical columns that have different defined category sets or order sequences raises a `TypeError`. Standardize the category types across both DataFrames before merging.
- **Forgetting observed=True in GroupBy**: In large datasets, grouping by multiple categorical columns without setting `observed=True` can cause memory errors because Pandas attempts to compute combinations for all categories, including those with zero occurrences.

---

## Best Practices

- Cast string columns to `category` if the number of unique values is less than 50% of the total row count.
- Define custom sequences using `pd.CategoricalDtype(..., ordered=True)` for ordinal data (like ratings, satisfaction scores, or sizes).
- Use `observed=True` inside `.groupby()` when grouping by categorical columns to exclude empty categories and speed up calculations.
- Modify category labels using `.cat.rename_categories()` or `.cat.add_categories()` to preserve the categorical data type.

---

## Worked Real-World Examples

### Worked Example 1: Corporate Performance Ordering

```python
import pandas as pd

# Employee roster
df_staff = pd.DataFrame({
    "Name": ["Aarav", "Neha", "Vikram", "Pooja"],
    "Rating": ["Exceeds", "Meets", "Exceeds", "Needs Improvement"]
})

# 1. Define custom performance levels
rating_type = pd.CategoricalDtype(
    categories=["Needs Improvement", "Meets", "Exceeds"],
    ordered=True
)

# 2. Cast performance column
df_staff["Rating"] = df_staff["Rating"].astype(rating_type)

# 3. Sort employees by performance rating descending
sorted_staff = df_staff.sort_values(by="Rating", ascending=False)

print("--- Employees Ranked by Performance ---")
print(sorted_staff)
```

### Output

```text
--- Employees Ranked by Performance ---
     Name             Rating
0   Aarav            Exceeds
2  Vikram            Exceeds
1    Neha              Meets
3   Pooja  Needs Improvement
```

---

### Worked Example 2: Optimizing Large Log Files

```python
import pandas as pd
import numpy as np

# Simulate large log file with repeated region tags
regions = ["North", "South", "East", "West"]
large_regions = np.random.choice(regions, size=100000)

df_logs = pd.DataFrame({"Region": large_regions})

# Calculate memory usage as object
mem_obj = df_logs["Region"].memory_usage(deep=True)

# Convert to category
df_logs["Region"] = df_logs["Region"].astype("category")
mem_cat = df_logs["Region"].memory_usage(deep=True)

print("--- Log Memory Footprints ---")
print("Object type (Bytes):", mem_obj)
print("Categorical type (Bytes):", mem_cat)
print(f"Memory reduction: {((mem_obj - mem_cat) / mem_obj) * 100:.2f}%")
```

### Output

```text
--- Log Memory Footprints ---
Object type (Bytes): 6200128
Categorical type (Bytes): 100416
Memory reduction: 98.38%
```

---

### Worked Example 3: Modifying Category Labels

```python
import pandas as pd

# Active categories
s_gender = pd.Series(["M", "F", "M", "F"], dtype="category")

# Rename category codes to descriptive labels
s_gender = s_gender.cat.rename_categories({"M": "Male", "F": "Female"})

print("--- Described Gender Categories ---")
print(s_gender)
```

### Output

```text
--- Described Gender Categories ---
0      Male
1    Female
2      Male
3    Female
dtype: category
Categories (2, object): ['Female', 'Male']
```

---

## Practice Questions

1. Explain how Pandas stores categorical data in memory under the hood.
2. Write a command to convert a string column `df['Size']` to an ordered categorical type with categories: `['S', 'M', 'L', 'XL']`.
3. Under what conditions does converting a string column to `category` increase memory usage?
4. Write a command to display the integer codes for a categorical Series `s`.
5. Explain the purpose and usage of the `observed=True` parameter inside `.groupby()`.
6. Write a command to add the category label `"Platinum"` to a categorical Series `s` using the `.cat` accessor.
7. What error occurs when you merge two DataFrames on categorical columns that have different defined category sets?
8. Write a script that reorders the categories of a categorical Series `s` to `['Low', 'Medium', 'High']` without changing the values.
9. Compare the execution speed of sorting an object Series versus a categorical Series with 1,000,000 rows.
10. Describe how to remove unused category levels from a categorical column after filtering rows.

---

## Mini Assignments

### Assignment 1: Customer Survey Satisfaction Rank
- Create a survey dataset with 10 records tracking `Respondent_ID` and `Satisfaction_Score` (containing values `"Dissatisfied"`, `"Neutral"`, `"Satisfied"`).
- Cast `Satisfaction_Score` to an ordered categorical type where `Dissatisfied < Neutral < Satisfied`.
- Sort the survey by satisfaction score descending and print the results.

### Assignment 2: E-commerce Product Size Ordering
- Create a sales DataFrame containing `OrderID`, `Product_Name`, and `Clothing_Size` (values like `S`, `M`, `L`, `XL` in unsorted rows).
- Convert `Clothing_Size` to an ordered categorical type.
- Group the sales by `Clothing_Size` and calculate transaction counts, ensuring the sizes are ordered correctly in the summary table.

### Assignment 3: CRM Database Memory Optimization
- Generate a customer log DataFrame containing `State` (50 states, highly duplicated strings across 10,000 rows) and `Signup_Date`.
- Calculate the memory footprint of the `State` column.
- Cast `State` to category, recalculate memory usage, and print the percentage memory reduction.

---

## Interview-Oriented Questions

- **Why does converting string columns to the `category` data type save memory?**
  - *Answer*: String columns store full text strings in memory for every row, which is inefficient. Categorical columns store unique strings once in a lookup table (categories) and represent each row value using an integer code (typically 8-bit or 16-bit). This reduces the memory footprint for columns with duplicate strings.
- **Explain the difference between nominal and ordinal categorical data in Pandas.**
  - *Answer*: Nominal categorical data contains categories with no logical order (e.g. colors, regions). Ordinal categorical data contains categories with an inherent logical order (e.g. Bronze < Silver < Gold). In Pandas, ordinal columns are defined using `pd.CategoricalDtype(..., ordered=True)`.
- **How can we identify and remove unused categories from a categorical column after filtering a DataFrame?**
  - *Answer*: After filtering, categories that are no longer present in the column remain in the defined categories lookup table. Use `df['Col'] = df['Col'].cat.remove_unused_categories()` to clean the categories table.
- **Why does applying string operations (like `.str.lower()`) to a categorical column convert it back to an object type?**
  - *Answer*: The `.str` accessor is designed for standard string objects. Applying string methods creates a new Series of standard string objects, stripping the categories lookup table and custom ordering. To rename categories, use `.cat.rename_categories()`.
- **What is the significance of the `observed` parameter in GroupBy operations involving categorical columns?**
  - *Answer*: By default, `observed=False` computes combinations for all defined categories, including those with zero occurrences in the grouped subset. Setting `observed=True` limits the output to categories that actually exist in the data, reducing memory usage and calculation times.

---

## Teaching Notes for This Chapter

- **Deconstruct Memory Layout**: Draw the split between categories and integer codes on a whiteboard to illustrate how categorical data is stored.
- **Showcase GroupBy performance**: Run a live demo comparing GroupBy calculations on a large DataFrame before and after categorical conversion.
- **Highlight sorting consistency**: Emphasize that custom categorical ordering ensures consistent sorting in reports, dashboards, and charts.

---

## Chapter Wrap-up Concepts Students Must Master

- Categorical data stores unique strings in a lookup table and represents each row value using an integer code.
- Convert string columns with repeating values to `category` to reduce memory usage and speed up calculations.
- Use `pd.CategoricalDtype(categories, ordered=True)` to define custom order sequences for ordinal data.
- Access categorical methods using the `.cat` accessor.
- Set `observed=True` in GroupBy operations to exclude empty categories.
- Avoid converting high-cardinality string columns (like `Email` or `UUID`) to `category` to prevent memory bloating.
