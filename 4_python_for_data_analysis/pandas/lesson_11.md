# Data Cleaning and Data Quality Workflows

## Lesson Overview

- This chapter focuses on building robust data quality pipelines. We explore identifying and removing duplicates, renaming axis elements, mapping and replacing specific data flags, and detecting statistical outliers.
- Raw data is notoriously messy. Inconsistencies like duplicate submissions, outdated column names, and extreme anomalies (outliers) degrade downstream statistical analyses and machine learning models.
- We will cover methods like `.duplicated()`, `.drop_duplicates()`, `.replace()`, `.rename()`, and apply statistical techniques (IQR and Z-score) to isolate and flag outliers.
- Mastering these data quality workflows is a prerequisite for generating reliable business intelligence reports.

## Learning Objectives

- Identify and drop duplicate rows or subset-specific duplicates using `.duplicated()` and `.drop_duplicates()`.
- Map and transform specific variables using `.replace()` with dictionaries or lists.
- Rename column headers and index labels systematically using `.rename()` with string functions or key maps.
- Implement the Interquartile Range (IQR) method and Z-score thresholds to detect and filter statistical outliers.
- Establish validation check routines to verify column bounds and data sanity.

---

## 1. Handling Duplicate Records

Duplicate data frequently arises from merged databases or multiple form submissions.

### Functions
- `.duplicated()`: Returns a boolean Series indicating if a row is a duplicate of a previous row.
- `.drop_duplicates()`: Returns a DataFrame with duplicate rows removed.

### Key Parameters
- `subset`: Only consider certain columns for identifying duplicates.
- `keep`: `'first'` (default, marks first occurrence as unique, others as duplicates), `'last'` (marks last occurrence as unique), or `False` (marks all occurrences of duplicate values as duplicates).

### Setup for Demonstration

```python
import pandas as pd

# Employee signup logs with duplicate entries
df_signup = pd.DataFrame({
    "UserID": ["U101", "U102", "U101", "U103", "U102"],
    "Name": ["Aarav", "Neha", "Aarav", "Pooja", "Neha"],
    "Platform": ["Web", "App", "Web", "Web", "Web"]
})

print("--- Master Signup Table ---")
print(df_signup)
```

### Output

```text
--- Master Signup Table ---
  UserID   Name Platform
0   U101  Aarav      Web
1   U102   Neha      App
2   U101  Aarav      Web
3   U103  Pooja      Web
4   U102   Neha      Web
```

---

### Applying Duplicate Checks

```python
# Identify duplicates (default keep='first')
print("--- Is Duplicate Row Mask ---")
print(df_signup.duplicated())

# Drop duplicates completely
unique_rows = df_signup.drop_duplicates()
print("\n--- Unique Rows (Complete Match) ---")
print(unique_rows)

# Drop duplicates checking UserID only, keeping the last signup
unique_userid = df_signup.drop_duplicates(subset=["UserID"], keep="last")
print("\n--- Unique UserID (Keep Last Signup) ---")
print(unique_userid)
```

### Output

```text
--- Is Duplicate Row Mask ---
0    False
1    False
2     True
3    False
4    False
dtype: bool

--- Unique Rows (Complete Match) ---
  UserID   Name Platform
0   U101  Aarav      Web
1   U102   Neha      App
3   U103  Pooja      Web
4   U102   Neha      Web

--- Unique UserID (Keep Last Signup) ---
  UserID   Name Platform
2   U101  Aarav      Web
3   U103  Pooja      Web
4   U102   Neha      Web
```

---

## 2. Renaming Axis Elements with `.rename()`

Rename column headers or row indexes to standardize names (e.g. converting to snake_case or removing spaces).

```python
df_messy = pd.DataFrame({
    "First Name": ["Aarav"],
    "age": [25],
    "SALARY_usd": [80000]
})

# Rename using a dictionary mapping
rename_dict = {
    "First Name": "first_name",
    "age": "age",
    "SALARY_usd": "salary_usd"
}

df_clean_cols = df_messy.rename(columns=rename_dict)
print("--- Renamed via Dictionary ---")
print(df_clean_cols)

# Rename dynamically using string functions
df_standardized = df_messy.rename(columns=str.lower)
print("\n--- Renamed via str.lower ---")
print(df_standardized)
```

### Output

```text
--- Renamed via Dictionary ---
  first_name  age  salary_usd
0      Aarav   25       80000

--- Renamed via str.lower ---
  first name  age  salary_usd
0      Aarav   25       80000
```

---

## 3. Detecting and Handling Outliers

Outliers are extreme data points that differ significantly from other observations. They can represent errors or valid anomalous events.

### Method A: The Interquartile Range (IQR) Method
The IQR method defines outliers as points that fall outside $[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$, where:
- $Q1$ (25th percentile): First quartile.
- $Q3$ (75th percentile): Third quartile.
- $IQR = Q3 - Q1$.

```python
import numpy as np

# Height data (in cm) with extreme outliers
df_height = pd.DataFrame({"Height": [160, 165, 170, 162, 168, 250, 40, 172]})

# Calculate quartiles
Q1 = df_height["Height"].quantile(0.25)
Q3 = df_height["Height"].quantile(0.75)
IQR = Q3 - Q1

# Calculate bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers_iqr = df_height[(df_height["Height"] < lower_bound) | (df_height["Height"] > upper_bound)]
print("IQR Bounds:", lower_bound, "to", upper_bound)
print("\n--- IQR Detected Outliers ---")
print(outliers_iqr)

# Clean dataset
df_height_clean = df_height[(df_height["Height"] >= lower_bound) & (df_height["Height"] <= upper_bound)]
print("\n--- Cleaned Height Table ---")
print(df_height_clean)
```

### Output

```text
IQR Bounds: 151.75 to 177.75

--- IQR Detected Outliers ---
   Height
5     250
6      40

--- Cleaned Height Table ---
   Height
0     160
1     165
2     170
3     162
4     168
7     172
```

---

### Method B: The Z-Score Method
The Z-Score indicates how many standard deviations a data point is from the mean. Ratios exceeding a threshold (typically $+3$ or $-3$) are classified as outliers.

$$Z = \frac{X - \mu}{\sigma}$$

```python
# Calculate mean and standard deviation
mean_val = df_height["Height"].mean()
std_val = df_height["Height"].std()

# Compute Z-score
df_height["Z_Score"] = (df_height["Height"] - mean_val) / std_val
print("--- Dataset with Z-Scores ---")
print(df_height)

# Filter where absolute Z-score is greater than 1.5 (using low threshold for demo)
outliers_z = df_height[df_height["Z_Score"].abs() > 1.5]
print("\n--- Z-Score Detected Outliers (Threshold > 1.5) ---")
print(outliers_z)
```

### Output

```text
--- Dataset with Z-Scores ---
   Height   Z_Score
0     160 -0.012586
1     165  0.071318
2     170  0.155221
3     162 -0.004195
4     168  0.121659
5     250  1.497746
6      40 -2.026264
7     172  0.188783

--- Z-Score Detected Outliers (Threshold > 1.5) ---
   Height   Z_Score
6      40 -2.026264
```

---

## Common Mistakes Students Make

- **Using `.drop_duplicates()` without checking indices**: Dropping duplicates preserves original index tags (e.g. indices 0, 1, 3 remain, while 2 is dropped). Failing to run `.reset_index(drop=True)` leads to index errors during subsequent alignments.
- **Forgetting parentheses inside outlier conditions**: Writing `df[(df['Val'] < lower) | (df['Val'] > upper)]` is correct. Writing `df[df['Val'] < lower | df['Val'] > upper]` throws parsing exceptions due to operator precedence.
- **In-place renaming syntax confusion**: Executing `df.rename(columns={'A': 'a'})` returns a copy and leaves the original DataFrame unchanged. Assign it back to a variable or use `inplace=True`.
- **Applying IQR outliers to skewed distributions without transformation**: Using standard symmetric IQR on heavily skewed distributions (like income or website traffic) results in identifying valid high-value points as outliers. Apply log transformations before statistical cleaning.

---

## Best Practices

- Always specify `keep='first'` or `keep='last'` explicitly when dropping duplicates to make your intent clear.
- Convert all column names to standard snake_case using `.rename(columns=str.lower)` and replacing spaces with underscores immediately after loading a dataset.
- Clean invalid placeholder markers (e.g., negative age inputs or non-numeric symbols) using `.replace()` before running statistical calculations.
- Use IQR limits rather than absolute Z-scores when data is not normally distributed, as IQR is robust to extreme outliers.

---

## Worked Real-World Examples

### Worked Example 1: Standardizing CRM Customer Signup Lists

```python
import pandas as pd

# 1. CRM Lead ingestion
crm_leads = pd.DataFrame({
    "Full Name": ["Aarav Patel", "Neha Sharma", "Aarav Patel", "Priya Sen"],
    "ContactNo": ["98765", "87654", "98765", "76543"],
    "City": ["MUMBAI", "delhi", "Mumbai", "Kolkata"]
})

# 2. Convert all column names to snake_case
crm_leads.columns = crm_leads.columns.str.replace(" ", "_").str.lower()

# 3. Clean city strings to title case
crm_leads["city"] = crm_leads["city"].str.title()

# 4. Drop duplicated contacts keeping the first record
cleaned_leads = crm_leads.drop_duplicates(subset=["contactno"], keep="first").reset_index(drop=True)

print("--- Cleaned CRM Contacts ---")
print(cleaned_leads)
```

### Output

```text
--- Cleaned CRM Contacts ---
     full_name contactno     city
0  Aarav Patel     98765   Mumbai
1  Neha Sharma     87654    Delhi
2    Priya Sen     76543  Kolkata
```

---

### Worked Example 2: Financial Sales Transactions Outlier Filter

```python
import pandas as pd

# Store transactions
sales_data = pd.DataFrame({
    "TxID": ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"],
    "Amount": [150, 200, 180, 220, 1500, 190, 15, 210]
})

# Calculate IQR bounds
Q1 = sales_data["Amount"].quantile(0.25)
Q3 = sales_data["Amount"].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Extract valid sales transactions
valid_sales = sales_data[sales_data["Amount"].between(lower_limit, upper_limit)]
anomalous_sales = sales_data[~sales_data["Amount"].between(lower_limit, upper_limit)]

print("--- Normal Sales Transactions ---")
print(valid_sales)

print("\n--- Flagged Outlier Transactions ---")
print(anomalous_sales)
```

### Output

```text
--- Normal Sales Transactions ---
  TxID  Amount
0   T1     150
1   T2     200
2   T3     180
3   T4     220
5   T6     190
7   T8     210

--- Flagged Outlier Transactions ---
  TxID  Amount
4   T5    1500
6   T7      15
```

---

### Worked Example 3: Value Mapping and Replacement

```python
import pandas as pd

df_survey = pd.DataFrame({
    "Respondent": [1, 2, 3, 4],
    "Preference": ["Yes", "No", "Maybe", "Y"]
})

# Normalize responses using dictionary mapping
mapping = {"Y": "Yes", "Maybe": "No"}
df_survey["Preference_Clean"] = df_survey["Preference"].replace(mapping)

print("--- Standardized Preference Survey ---")
print(df_survey)
```

### Output

```text
--- Standardized Preference Survey ---
   Respondent Preference Preference_Clean
0           1        Yes              Yes
1           2         No               No
2           3      Maybe               No
3           4          Y              Yes
```

---

## Practice Questions

1. Explain the differences between the `.duplicated()` and `.drop_duplicates()` methods in Pandas.
2. How does the `keep=False` parameter change the behavior of duplicate row detection?
3. Write a command to rename column `Old_Col` to `New_Col` and column `Sales_Amt` to `sales_amt` in-place.
4. Calculate the IQR mathematically for the following data array: `[10, 12, 15, 18, 20, 22, 25, 100]`.
5. Why is the median used instead of the mean when calculating the central tendency of outlier-prone distributions?
6. Write a script that checks for duplicate rows based strictly on columns `Product_ID` and `Supplier_ID`.
7. What is the Z-score value of a data point that is equal to the sample mean?
8. Compose a `.rename()` statement using a lambda function to add the prefix `"sensor_"` to all columns in a DataFrame.
9. How can you replace multiple distinct placeholder values (e.g. `9999` and `"-999"`) with standard `NaN`?
10. Define the upper and lower limits of the IQR outlier detection threshold.

---

## Mini Assignments

### Assignment 1: Student Database Deduplication
- Create a student enrollment dataset of 8 entries containing `Roll_No`, `Student_Name`, `Course_Code`, and `Registration_Date` (some rows must have identical `Roll_No` and `Course_Code`).
- Identify and print rows that represent duplicate course registrations for the same student.
- Drop duplicate registrations, keeping the first registration date, and reset the index.

### Assignment 2: E-commerce Price Outlier Purge
- Create a dataset of 12 online store products with columns: `Item_ID`, `Group_Name`, and `Price`. Introduce two products with extreme prices (e.g., $1.00 and $10,000.00).
- Calculate the IQR bounds for the price column.
- Filter out price outliers, output the cleaned catalog, and display the flagged outlier items.

### Assignment 3: Column Header Sanitisation
- Scaffold a raw DataFrame with headers: `"User ID"`, `"User AGE"`, `"Join Date "`, and `" ACCOUNT_BALANCE_usd"`.
- Clean the column headers programmatically: strip leading/trailing spaces, convert to lowercase, and replace internal spaces with underscores.

---

## Interview-Oriented Questions

- **How do you choose between the IQR method and the Z-score method for outlier detection?**
  - *Answer*: Use the Z-score method when the data is normally distributed, as it depends on the mean and standard deviation. Use the IQR method when the data is skewed or has non-normal distributions, as it is based on percentiles (medians), which are robust to outliers.
- **Explain the performance difference between `df.drop_duplicates()` and checking unique indices manually.**
  - *Answer*: `df.drop_duplicates()` is highly optimized in C under the hood, hashing row values to find matches. Manual loops or index checking is slower and has higher memory overhead.
- **How does the `keep` parameter affect the return footprint of `df.duplicated()`?**
  - *Answer*: If `keep='first'`, the first occurrence of a row is marked `False` (not duplicate) and subsequent occurrences are marked `True`. If `keep='last'`, the last occurrence is marked `False` and previous ones are `True`. If `keep=False`, all occurrences of duplicate values are marked `True`.
- **What is the mathematical definition of a Z-score, and what do positive/negative Z-scores signify?**
  - *Answer*: A Z-score is defined as $Z = (X - \mu) / \sigma$. A positive Z-score indicates the data point is above the mean, while a negative Z-score indicates it is below the mean.
- **Can we rename both the index labels and column names in a single `.rename()` invocation?**
  - *Answer*: Yes, you can pass both `index={...}` and `columns={...}` mapping dictionaries to a single `.rename()` method call.

---

## Teaching Notes for This Chapter

- **Contrast Duplication Methods**: Use a small table to illustrate the difference between row deduplication and subset deduplication.
- **Explain Outlier Sensitivity**: Highlight how a single extreme outlier can inflate the mean and standard deviation, reducing the accuracy of Z-score detection.
- **Emphasize Index Resetting**: Always remind students to use `.reset_index(drop=True)` after dropping duplicates to prevent issues with index alignments.

---

## Chapter Wrap-up Concepts Students Must Master

- Use `.duplicated()` to detect duplicate rows and `.drop_duplicates()` to remove them.
- `.rename()` standardizes index labels and column names.
- Outlier detection identifies extreme values using statistical methods: Z-score (for normal distributions) and IQR (for skewed distributions).
- Replace specific value placeholders (e.g., `"N/A"` or `-999`) using `.replace()`.
- Reset DataFrame indexes post-cleaning to maintain sequential numbering.
