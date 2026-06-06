# Working with Text and String Data

## Lesson Overview

- This chapter covers text preprocessing using Pandas' vectorized string operations via the `.str` accessor.
- Text data is often messy, containing leading/trailing whitespace, inconsistent casing, and embedded sub-strings. If uncleaned, these text fields prevent database indexing, lead to duplicate category mappings, and break text-based analytical queries.
- We will cover standard operations like `.str.strip()`, `.str.lower()`, `.str.split()`, `.str.contains()`, and explore advanced regular expression parsing using `.str.extract()` and `.str.findall()`.
- Mastering the `.str` accessor allows you to clean text fields and extract structured features from unstructured string inputs.

## Learning Objectives

- Apply vectorized string functions (casing, trimming, replacement) using the `.str` accessor.
- Isolate characters or sub-strings dynamically using vectorized slicing (`.str[...]`).
- Split text strings into lists or multiple columns using `.str.split()`.
- Detect sub-strings or text patterns using `.str.contains()` with regular expressions.
- Extract structured values from text using regex capture groups in `.str.extract()`.
- Understand how string methods propagate missing (`NaN`) values.

---

## Vectorized String Operations and the `.str` Accessor

To apply string methods (like `.lower()` or `.strip()`) element-wise across a Series, you **must** prefix the method with the **`.str`** accessor. Attempting to call string methods directly on a Series (e.g. `s.lower()`) raises an `AttributeError`.

### Setup for Demonstration

```python
import pandas as pd

# Customer registration details
df_users = pd.DataFrame({
    "UserID": ["U_1", "U_2", "U_3", "U_4"],
    "Name": [" Aarav Patel ", "neha sharma", "vikram.rathore", "Pooja Sen"],
    "Email": ["aarav_patel@test.com", "neha.sharma@domain.co.in", np.nan, "pooja.sen@test.org"]
})

print("--- Master Users Table ---")
print(df_users)
```

### Output

```text
--- Master Users Table ---
  UserID            Name                     Email
0    U_1    Aarav Patel       aarav_patel@test.com
1    U_2     neha sharma  neha.sharma@domain.co.in
2    U_3  vikram.rathore                       NaN
3    U_4       Pooja Sen        pooja.sen@test.org
```

---

### 1. Basic Cleaning: Trimming, Casing, and Slicing

```python
# Trim whitespace and capitalize name
df_users["Name_Clean"] = df_users["Name"].str.strip().str.title()

# Vectorized slicing to extract the first 3 characters of names
df_users["Name_Prefix"] = df_users["Name_Clean"].str[:3]

print("--- Cleaned and Sliced Names ---")
print(df_users[["Name", "Name_Clean", "Name_Prefix"]])
```

### Output

```text
--- Cleaned and Sliced Names ---
             Name      Name_Clean Name_Prefix
0    Aarav Patel      Aarav Patel         Aar
1     neha sharma     Neha Sharma         Neh
2  vikram.rathore  Vikram.Rathore         Vik
3       Pooja Sen       Pooja Sen         Poo
```

---

### 2. Splitting Strings into Multiple Columns

Use `.str.split()` to divide strings based on a delimiter. Set `expand=True` to return a DataFrame of separate columns instead of a Series of lists.

```python
# Split Name_Clean by spaces
split_names = df_users["Name_Clean"].str.split(" ", expand=True)

# Rename the columns
split_names.columns = ["First_Name", "Last_Name"]

print("--- Split Name DataFrame ---")
print(split_names)
```

### Output

```text
--- Split Name DataFrame ---
  First_Name      Last_Name
0      Aarav          Patel
1       Neha         Sharma
2     Vikram  Rathore  # split by dot was not requested, so dot remained as a single string
3      Pooja            Sen
```
*Note: If a row doesn't contain the delimiter (e.g. Vikram.Rathore lacks a space), the second column is filled with None.*

---

## 3. Pattern Matching and Sub-string Checks

Use `.str.contains()` to search for sub-strings. By default, `.str.contains` supports regular expressions.

```python
# Check if email contains 'test'
# na=False keeps the output as boolean (replacing NaNs with False)
df_users["Is_Test_Email"] = df_users["Email"].str.contains("test", na=False)

print("--- Test Email Filter ---")
print(df_users[["Email", "Is_Test_Email"]])
```

### Output

```text
--- Test Email Filter ---
                      Email  Is_Test_Email
0      aarav_patel@test.com           True
1  neha.sharma@domain.co.in          False
2                       NaN          False
3        pooja.sen@test.org           True
```

---

## 4. Extracting Structured Data using Regex Capture Groups

Use `.str.extract()` with regular expression capture groups `()` to extract specific text structures (e.g., usernames, domain names, or phone codes) into separate columns.

```python
# Extract email username and domain name
# Regex format: ([^@]+) matching username, followed by @, and ([^@]+) matching domain
email_extract = df_users["Email"].str.extract(r"([^@]+)@([^@]+)")
email_extract.columns = ["Username", "Domain"]

print("--- Extracted Email Features ---")
print(email_extract)
```

### Output

```text
--- Extracted Email Features ---
      Username          Domain
0  aarav_patel        test.com
1  neha.sharma    domain.co.in
2          NaN             NaN
3    pooja.sen        test.org
```

---

## 5. String Manipulation & Missing Values (NaN)

Vectorized string operations in Pandas preserve missing values: if the input is `NaN`, the output of the string operation will also be `NaN`.

```python
# Appending text to Email column
appended_email = df_users["Email"].str.upper()
print("--- Uppercase Emails ---")
print(appended_email)
```

### Output

```text
--- Uppercase Emails ---
0        AARAV_PATEL@TEST.COM
1    NEHA.SHARMA@DOMAIN.CO.IN
2                         NaN
3          POOJA.SEN@TEST.ORG
Name: Email, dtype: object
```
*Note: The missing value at index 2 remained `NaN` instead of raising a TypeError.*

---

## Common Mistakes Students Make

- **Omitting the `.str` accessor**: Writing `df['Col'].lower()` raises an `AttributeError: 'Series' object has no attribute 'lower'`. Always write `df['Col'].str.lower()`.
- **Unexpected NaN behavior in filters**: Running `df[df['Email'].str.contains('test')]` throws an error (`ValueError: Cannot mask with non-boolean array`) if the column contains `NaN` values, because the string check returns `NaN` instead of `True` or `False`. Always pass `na=False` or `na=True` inside `.str.contains()`.
- **Forgetting `expand=True` on splits**: Running `df['Col'].str.split(',')` returns a Series where each row is a Python list. To get a DataFrame of separate columns, use `df['Col'].str.split(',', expand=True)`.
- **Excessive compilation overhead with Regex**: When using `.str.contains()` for simple string lookups, students often forget that regex is enabled by default. If search strings contain regex symbols (like `?`, `*`, or `.`), set `regex=False` to prevent compile errors.

---

## Best Practices

- Always use the `na=False` parameter inside `.str.contains()` to ensure boolean outputs for filtering.
- Use `regex=False` for simple, exact sub-string lookups to speed up execution.
- Standardize on `.str.strip()` to remove leading and trailing spaces immediately after loading text data.
- Leverage `.str.extract()` with named regex capture groups to extract clean columns directly.

---

## Worked Real-World Examples

### Worked Example 1: Sanitizing CRM Product IDs

```python
import pandas as pd

# Raw product logs
products = pd.DataFrame({
    "RawID": [" prod_A101-Tech ", "PROD_B202-Home", "prod_c303-apparel "],
    "Price": [120, 450, 80]
})

# 1. Strip whitespace and convert to lowercase
products["RawID"] = products["RawID"].str.strip().str.lower()

# 2. Extract product code and category using regex
extracted_features = products["RawID"].str.extract(r"prod_([a-z0-9]+)-([a-z]+)")
extracted_features.columns = ["Product_Code", "Category"]

# 3. Concatenate clean columns
products = pd.concat([products, extracted_features], axis=1)

print("--- Cleaned Products DataFrame ---")
print(products)
```

### Output

```text
--- Cleaned Products DataFrame ---
                RawID  Price Product_Code  Category
0     prod_a101-tech    120         a101      tech
1     prod_b202-home    450         b202      home
2  prod_c303-apparel     80         c303   apparel
```

---

### Worked Example 2: Parsing Names and Handling Nulls

```python
import pandas as pd
import numpy as np

# Roster of customers
roster = pd.DataFrame({
    "ID": [101, 102, 103],
    "FullName": ["Amit Kumar", np.nan, "Karan Johar"]
})

# Extract first name, handling null values safely
roster["First_Name"] = roster["FullName"].str.split(" ").str[0]

print("--- Roster with First Names ---")
print(roster)
```

### Output

```text
--- Roster with First Names ---
    ID     FullName First_Name
0  101   Amit Kumar       Amit
1  102          NaN        NaN
2  103  Karan Johar      Karan
```

---

### Worked Example 3: Finding Anomalous Sensor Errors

```python
import pandas as pd

# Sensor log entries
sensor_logs = pd.DataFrame({
    "LogID": ["L1", "L2", "L3", "L4"],
    "Message": ["TEMP: 45C - Status OK", "TEMP: 85C - ERROR: OVERHEAT", "HUMID: 12% - Status OK", "TEMP: ERROR: SENSOR_OFF"]
})

# Filter logs containing "ERROR" in the message
error_logs = sensor_logs[sensor_logs["Message"].str.contains("ERROR", case=True)]

print("--- Flagged Error Logs ---")
print(error_logs)
```

### Output

```text
--- Flagged Error Logs ---
  LogID                      Message
1    L2  TEMP: 85C - ERROR: OVERHEAT
3    L4      TEMP: ERROR: SENSOR_OFF
```

---

## Practice Questions

1. Explain the purpose and necessity of the `.str` accessor in Pandas Series operations.
2. Write a command to convert a string Series `s` to uppercase, replace all hyphens with underscores, and strip whitespaces.
3. How does `.str.split()` differ in output when `expand=True` is specified versus when it is omitted?
4. Write a command to filter a DataFrame `df` where the column `Product_Desc` contains the word `"Premium"` case-insensitively, handling missing values.
5. What is the return value of calling a string method (like `.str.upper()`) on a row where the entry is `NaN`?
6. Write a regular expression for `.str.extract()` to extract the phone area code from strings in the format `"(123) 456-7890"`.
7. How do you perform vectorized string slicing to extract the last 4 characters of a Series of strings?
8. Explain the difference between `.str.contains()` and `.str.match()`.
9. Write a command to count the number of characters in each string of a Series, returning `0` if the value is missing.
10. Describe how you would extract multiple non-overlapping occurrences of a pattern from a text column.

---

## Mini Assignments

### Assignment 1: Customer Username and Domain Sanitisation
- Create a client dataset with 8 records containing `Client_ID`, `Full_Name`, and `Contact_Email` (include some missing email entries).
- Using string operations, extract the email username and domain into separate columns.
- Standardize name casing to title case and strip any trailing dot symbols from names.

### Assignment 2: URL Query Parameter Extractor
- Create a DataFrame containing 5 website access logs with a `URL` column (e.g. `"/shop/product?id=101&category=tech"`, `"/shop/product?id=202&category=home"`).
- Use `.str.extract()` and regex capture groups to extract the `id` and `category` parameters into separate columns.

### Assignment 3: Inventory Log Audit
- Load a product description list containing text comments.
- Search for descriptions containing keywords `"fragile"`, `"heavy"`, or `"hazardous"`.
- Add a boolean flag column `Special_Handling` that is `True` if any of these keywords are matched, and `False` otherwise (handling missing values).

---

## Interview-Oriented Questions

- **Why does omitting `na=False` inside `.str.contains()` throw an error when used as a filter mask?**
  - *Answer*: If a column contains `NaN` values, `.str.contains()` returns `NaN` for those entries. A boolean mask used to filter a DataFrame must contain exclusively `True` or `False` values; the presence of `NaN` makes the mask ambiguous, causing Pandas to raise a `ValueError`. Passing `na=False` replaces all `NaN` results with `False`.
- **Explain the difference between `.str.extract()` and `.str.extractall()`.**
  - *Answer*: `.str.extract()` searches for the first match of the regular expression capture groups in each row and returns a DataFrame with one column per group. `.str.extractall()` finds *all* matching occurrences of the pattern in each row, returning a MultiIndexed DataFrame where the second level of the index tracks the match occurrence number.
- **How can we disable regular expression parsing in `.str.contains()` for faster performance?**
  - *Answer*: Pass the parameter `regex=False` to `.str.contains()`. This forces Pandas to perform a literal string search, which is faster and avoids syntax issues with characters like `*` or `?`.
- **How does the `.str.cat()` method perform string concatenation, and how does it handle missing values?**
  - *Answer*: `.str.cat()` concatenates a Series of strings with another Series, list, or scalar, using an optional delimiter. By default, if any value in the concatenation is `NaN`, the result is `NaN`. Pass `na_rep="Unknown"` to replace missing values with a string placeholder.
- **Can we use vectorized string methods on a DataFrame directly?**
  - *Answer*: No, the `.str` accessor is only available on Pandas Series and Index objects. To apply string methods to an entire DataFrame, use `df.applymap(lambda x: x.lower() if isinstance(x, str) else x)` or map columns individually.

---

## Teaching Notes for This Chapter

- **Deconstruct Regular Expressions**: Walk students through basic regex patterns (e.g. `[a-zA-Z]`, `\d+`, and `[^@]+`) to build confidence before writing extraction statements.
- **Showcase the expand parameter**: Illustrate the output of `.str.split()` with and without `expand=True` on a whiteboard.
- **Reinforce Missing Data Safety**: Remind students that string operations do not crash on missing data; instead, they propagate `NaN`s, which is clean but can cause issues with filtering masks.

---

## Chapter Wrap-up Concepts Students Must Master

- Use the `.str` accessor to apply string methods element-wise to Series.
- Vectorized operations propagate `NaN` values, returning `NaN` for any missing inputs.
- Use `na=False` inside `.str.contains()` when filtering DataFrames to avoid boolean mask errors.
- Use `.str.split(expand=True)` to split strings into separate columns.
- Use `.str.extract()` with regex capture groups to extract structured features from text fields.
